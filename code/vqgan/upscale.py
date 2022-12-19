import os.path as osp
import glob
import cv2
import numpy as np
import torch
from ESRGAN import RRDBNet_arch as arch
import requests
import imageio
import requests

parser = argparse.ArgumentParser(description="Upscale image")

parser.add_argument("--model_path", type=str, help="Path to RRDB model", required=True)
parser.add_argument(
        "--input", type=str, help="Path to input image", required=True
)
parser.add_argument('--output', type=str, help="Path to output image", required=True)

device = torch.device('cuda:0' if torch.cuda_is_available() else 'cpu')

model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(parser.model_path), strict=True)
model.eval()
model = model.to(device)

base = osp.splitext(osp.basename(parser.input))[0]
# read images
img = cv2.imread(parser.input, cv2.IMREAD_COLOR)
img = img * 1.0 / 255.0
img = torch.from_numpy(np.transpose(img[:, :, [0, 1, 2]], (2, 0, 1))).float()
img_LR = img.unsqueeze(0)
img_LR = img_LR.to(device)

with torch.no_grad():
    output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
output = (output * 255.0).round()
imageio.imwrite(parser.output, output.astype(np.uint8))
