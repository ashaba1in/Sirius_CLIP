import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF

import argparse
from CLIP import clip
from PIL import Image
import numpy as np
from helpers import load_vqgan_model, MakeCutouts, parse_prompt, Prompt, resize_image, vector_quantize, clamp_with_grad


class VQGANWithCLIP:
    def __init__(self, args: argparse.Namespace, device=None):
        self.args = args
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        print("Using device:", device)
        if args.prompts:
            print("Using text prompt:", args.prompts)
        if args.image_prompts:
            print("Using image prompts:", args.image_prompts)
        if args.seed is None:
            seed = torch.seed()
        else:
            seed = args.seed
        torch.manual_seed(seed)
        print("Using seed:", seed)

        self.model = load_vqgan_model(args.vqgan_config, args.vqgan_checkpoint).to(self.device)
        self.perceptor = (
            clip.load(args.clip_model, jit=False)[0].eval().requires_grad_(False).to(self.device)
        )

        cut_size = self.perceptor.visual.input_resolution
        e_dim = self.model.quantize.e_dim
        f = 2 ** (self.model.decoder.num_resolutions - 1)
        self.make_cutouts = MakeCutouts(cut_size, args.cutn, cut_pow=args.cut_pow)
        n_toks = self.model.quantize.n_e
        toksX, toksY = args.size[0] // f, args.size[1] // f
        sideX, sideY = toksX * f, toksY * f
        self.z_min = self.model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
        self.z_max = self.model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]

        if self.args.init_image:
            pil_image = Image.open(args.init_image).convert("RGB")
            pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
            self.z, *_ = self.model.encode(TF.to_tensor(pil_image).to(self.device).unsqueeze(0) * 2 - 1)
        else:
            one_hot = F.one_hot(
                torch.randint(n_toks, [toksY * toksX], device=self.device), n_toks
            ).float()
            self.z = one_hot @ self.model.quantize.embedding.weight
            self.z = self.z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2)
        self.z_orig = self.z.clone()
        self.z.requires_grad_(True)
        self.opt = optim.Adam([self.z], lr=args.step_size)

        self.normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
        )

        self.pMs = []

        for prompt in self.args.prompts:
            txt, weight, stop = parse_prompt(prompt)
            embed = self.perceptor.encode_text(clip.tokenize(txt).to(self.device)).float()
            self.pMs.append(Prompt(embed, weight, stop).to(self.device))

        for prompt in args.image_prompts:
            path, weight, stop = parse_prompt(prompt)
            img = resize_image(Image.open(path).convert("RGB"), (sideX, sideY))
            batch = self.make_cutouts(TF.to_tensor(img).unsqueeze(0).to(self.device))
            embed = self.perceptor.encode_image(self.normalize(batch)).float()
            self.pMs.append(Prompt(embed, weight, stop).to(self.device))

        for seed, weight in zip(args.noise_prompt_seeds, args.noise_prompt_weights):
            gen = torch.Generator().manual_seed(seed)
            embed = torch.empty([1, self.perceptor.visual.output_dim]).normal_(generator=gen)
            self.pMs.append(Prompt(embed, weight).to(self.device))

    def synth(self):
        z_q = vector_quantize(self.z.movedim(1, 3), self.model.quantize.embedding.weight).movedim(
            3, 1
        )
        return clamp_with_grad(self.model.decode(z_q).add(1).div(2), 0, 1)

    @torch.no_grad()
    def checkin(self, losses):
        losses_str = ", ".join(f"{loss.item():g}" for loss in losses)
        # tqdm.write(f'i: {i}, loss: {sum(losses).item():g}, losses: {losses_str}')
        out = self.synth()
        TF.to_pil_image(out[0].cpu()).save("progress.png")

    def ascend_txt(self, i, save_file=False):
        out = self.synth()
        iii = self.perceptor.encode_image(self.normalize(self.make_cutouts(out))).float()

        result = []

        if self.args.init_weight:
            result.append(F.mse_loss(self.z, self.z_orig) * self.args.init_weight / 2)

        for prompt in self.pMs:
            result.append(prompt(iii))
        img = np.array(
            out.mul(255).clamp(0, 255)[0].cpu().detach().numpy().astype(np.uint8)
        )[:, :, :]
        img = np.transpose(img, (1, 2, 0))
        if save_file:
            filename = f"result{i}.png"
            Image.fromarray(img).save(filename)
        return result

    def train_step(self, i):
        self.opt.zero_grad()
        lossAll = self.ascend_txt(i)
        if i % self.args.display_freq == 0:
            self.checkin(lossAll)
        loss = sum(lossAll)
        loss.backward()
        self.opt.step()
        with torch.no_grad():
            self.z.copy_(self.z.maximum(self.z_min).minimum(self.z_max))

    def train(self):
        try:
            i = 0
            while True:
                self.train_step(i)
                if i == self.args.max_iterations:
                    break
                i += 1
            self.ascend_txt(i, True)
        except KeyboardInterrupt:
            pass
