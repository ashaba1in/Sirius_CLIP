import os.path as osp
import glob
import cv2
import numpy as np
import torch
# ESRGAN must be in the same directory as this scrpit
from ESRGAN import RRDBNet_arch as arch
import requests
import imageio
import requests
import argparse

parser = argparse.ArgumentParser(description="Upscale image")

parser.add_argument("--model_path", type=str, help="Path to RRDB model", required=True)
parser.add_argument(
        "--input", type=str, help="Path to input image", required=True
)
parser.add_argument('--output', type=str, help="Path to output image", required=True)

params = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(params.model_path), strict=True)
model.eval()
model = model.to(device)

base = osp.splitext(osp.basename(params.input))[0]
# read images
img = cv2.imread(params.input, cv2.IMREAD_COLOR)
img = img * 1.0 / 255.0
img = torch.from_numpy(np.transpose(img[:, :, [0, 1, 2]], (2, 0, 1))).float()
img_LR = img.unsqueeze(0)
img_LR = img_LR.to(device)

with torch.no_grad():
    output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
output = (output * 255.0).round()
imageio.imwrite(params.output, output.astype(np.uint8))
