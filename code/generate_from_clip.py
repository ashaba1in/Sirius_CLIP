import os
os.sys.path.append('taming_transformers')
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from taming.models.vqgan import VQModel
from CLIP import clip

import math
import torch
from torch import nn
import torch.nn.functional as F
import kornia.augmentation as K

from torchvision import transforms
from torchvision.transforms import functional as TF

import matplotlib.pyplot as plt
from PIL import Image

import os

from tqdm.auto import tqdm

from IPython.display import clear_output

class ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward

    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)

replace_grad = ReplaceGrad.apply


class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(input)
        return input.clamp(min, max)

    @staticmethod
    def backward(ctx, grad_in):
        input, = ctx.saved_tensors
        return grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0), None, None

clamp_with_grad = ClampWithGrad.apply

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

normalize = transforms.Normalize(
    mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
)

def sinc(x):
    return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))


def lanczos(x, a):
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x/a), x.new_zeros([]))
    return out / out.sum()


def ramp(ratio, width):
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1:-1]
 

def resample(input, size, align_corners=True):
    n, c, h, w = input.shape
    dh, dw = size
 
    input = input.contiguous().view([n * c, 1, h, w])
 
    if dh < h:
        kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = F.pad(input, (0, 0, pad_h, pad_h), 'reflect')
        input = F.conv2d(input, kernel_h[None, None, :, None])
 
    if dw < w:
        kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = F.pad(input, (pad_w, pad_w, 0, 0), 'reflect')
        input = F.conv2d(input, kernel_w[None, None, None, :])
 
    input = input.view([n, c, h, w])
    return F.interpolate(input, size, mode='bicubic', align_corners=align_corners)


class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn=32, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.augs = nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            # K.RandomSolarize(0.01, 0.01, p=0.7),
            K.RandomSharpness(0.3,p=0.4),
            K.RandomAffine(degrees=30, translate=0.1, p=0.8, padding_mode='border'),
            K.RandomPerspective(0.2, p=0.4),
            # K.ColorJitter(hue=0.01, saturation=0.01, p=0.7)
        )
        self.noise_fac = 0.1

 
    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
        batch = self.augs(torch.cat(cutouts, dim=0))
        
        if self.noise_fac:
            facs = batch.new_empty([len(batch), 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)
        return batch
    
# model = load_vqgan_model(args.vqgan_config, args.vqgan_checkpoint).to(device)
perceptor, preprocess = clip.load('ViT-B/32', device=device)
perceptor = perceptor.eval().requires_grad_(False)

cut_size = perceptor.visual.input_resolution
make_cutouts = MakeCutouts(cut_size)

def load_vqgan_model(config_path, checkpoint_path):
    config = yaml.load(open(config_path, 'r'), Loader)

    model = VQModel(**config['model']['params'])
    model.eval().requires_grad_(False)
    model.init_from_ckpt(checkpoint_path)

    del model.loss
    return model

vqgan = load_vqgan_model('vqgan/config.yaml', 'vqgan/model.ckpt').to(device)

def vector_quantize(x, codebook):
    # dist [bs, size, size, n_tokens]: (x - codebook)^2 = x^2 + codebook^2 - 2 * x * codebook
    dist = x.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * x @ codebook.T
    indices = dist.argmin(-1)
    x_q = F.one_hot(indices, codebook.shape[0]).to(dist.dtype) @ codebook
    return replace_grad(x_q, x)

# constants
IMG_SIZE = 512

N_TOKENS = vqgan.quantize.n_e
EMB_DIM = vqgan.quantize.e_dim
f = 2**(vqgan.decoder.num_resolutions - 1)
TOKS_X = TOKS_Y = IMG_SIZE // f

# PROMPT = 'white cat'
#PROMPT = 'victorian house on a hill, picasso style'

def tv_loss(img):
    pixel_dif1 = img[:, :, 1:, :] - img[:, :, :-1, :]
    pixel_dif2 = img[:, :, :, 1:] - img[:, :, :, :-1]

    res1 = pixel_dif1.abs()
    res2 = pixel_dif2.abs()

    reduce_axes = (-3, -2, -1)
    res1 = res1.float().mean(dim=reduce_axes)
    res2 = res2.float().mean(dim=reduce_axes)
            
    return res1 + res2

def generate(promt):
    prompt_encoding = perceptor.encode_text(clip.tokenize(promt).to(device)).float()
    n_images = 3

    one_hot = F.one_hot(torch.randint(N_TOKENS, [n_images, 16 * 16], device=device), N_TOKENS).float()
    z = one_hot @ vqgan.quantize.embedding.weight
    z = z.view([-1, 16, 16, EMB_DIM]).permute(0, 3, 1, 2).requires_grad_(True)
    optimizer = torch.optim.Adam([z], lr=0.1337)
    out = []
    for i in tqdm(range(300)):
        z_q = vector_quantize(z.movedim(1, 3), vqgan.quantize.embedding.weight).movedim(3, 1)
        out = (vqgan.decode(z_q) + 1) / 2
        back_encoding = perceptor.encode_image(normalize(make_cutouts(F.interpolate(out, 300)))).float()

        input_normed = F.normalize(back_encoding.unsqueeze(1) + 1e-6, dim=2)
        embed_normed = F.normalize(prompt_encoding.unsqueeze(0) + 1e-6, dim=2)
        dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)

        # loss = tv_loss(out).mean()
        loss = dists.mean() # + tv_loss(out).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    img = []
    for i in range(n_images):
        img.append(out[i].detach().cpu().float().permute(1, 2, 0))
    return img