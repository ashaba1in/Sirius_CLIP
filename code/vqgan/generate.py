import argparse
from model import VQGANWithCLIP

parser = argparse.ArgumentParser(description="Generate image from prompt")

parser.add_argument("--seed", type=int, default=-1, help="Seed for generation", required=True)
parser.add_argument(
    "--display_frequency", type=int, default=50, help="Frequency of model result save", required=True
)
parser.add_argument("--prompts", type=str, help="Promp to generate images from", required=True)
parser.add_argument("--width", type=int, default=200, help="Width of output image")
parser.add_argument("--height", type=int, default=200, help="Height of output image")
parser.add_argument(
    "--clip_model", type=str, default="ViT-B/32", help="CLIP model name"
)
parser.add_argument(
    "--vqgan_model",
    type=str,
    default="vqgan_imagenet_f16_16384",
    help="VQGAN model name",
)
parser.add_argument(
    "--initial_image", type=str, default="", help="Image to generate from"
)
parser.add_argument(
    "--target_images", type=str, default="", help="Image to redact"
)
parser.add_argument(
    "--max_iterations", type=int, default=300, help="Max amount of iterations to teach model"
)
parser.add_argument(
    "--vq_init_weight", type=float, default=0.0, help="VQGAN initial weights"
)
parser.add_argument(
    "--vq_step_size", type=float, default=0.1, help="VQGAN learning rate for Adam"
)
parser.add_argument("--vq_cutn", type=int, default=64, help="VQGAN amount of crops")
parser.add_argument("--vq_cutpow", type=float, default=1.0, help="VQGAN images augmentation")

params = parser.parse_args()


#model_names = {
    #"vqgan_imagenet_f16_16384": "ImageNet 16384",
    #"vqgan_imagenet_f16_1024": "ImageNet 1024",
    #"wikiart_1024": "WikiArt 1024",
    #"wikiart_16384": "WikiArt 16384",
    #"coco": "COCO-Stuff",
    #"faceshq": "FacesHQ",
    #"sflckr": "S-FLCKR",
#}
#model_name = model_names[params.vqgan_model]

if params.seed == -1:
    params.seed = None
if params.initial_image == "None":
    params.initial_image = None
if params.target_images == "None" or not params.target_images:
    params.target_images = []
else:
    params.target_images = params.target_images.split("|")
    params.target_images = [image.strip() for image in params.target_images]

if params.initial_image or params.target_images != []:
    input_images = True

params.prompts = [frase.strip() for frase in params.prompts.split("|")]
if params.prompts == [""]:
    prompts = []


args = argparse.Namespace(
    prompts=params.prompts,
    image_prompts=params.target_images,
    noise_prompt_seeds=[],
    noise_prompt_weights=[],
    size=[params.width, params.height],
    init_image=params.initial_image,
    init_weight=0.0,
    clip_model=params.clip_model,
    vqgan_config=f"models/{params.vqgan_model}.yaml",
    vqgan_checkpoint=f"models/{params.vqgan_model}.ckpt",
    step_size=params.vq_step_size,
    cutn=params.vq_cutn,
    cut_pow=params.vq_cutpow,
    display_freq=params.display_frequency,
    seed=params.seed,
    max_iterations=params.max_iterations
)

model = VQGANWithCLIP(args)
model.train()
