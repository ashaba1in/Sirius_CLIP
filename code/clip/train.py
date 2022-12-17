from tqdm.autonotebook import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt
import tqdm
from model import create_model, device
import torch

image_path = 'drive/MyDrive/Sirius_CLIP/Images'
captions_path = 'drive/MyDrive/Sirius_CLIP/captions.txt'
image_size = 224
num_workers = 1
epochs=10
batch_size=64

class Dataset:
    def __init__(self, images_path, captions_path, transform=None):
        self.captions = pd.read_csv(captions_path)
        self.images_path = images_path
        self.transform = transform

    def __getitem__(self, idx):
        img_path, caption = self.captions.iloc[idx]
        img = Image.open(os.path.join(self.images_path, img_path))
        w, h = img.size

        # print([max(w, h) - w, max(w, h) - h])
        pad = torchvision.transforms.Pad([max(w, h) - w, max(w, h) - h], fill=0, padding_mode='constant')
        img = pad(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, caption

    def __len__(self):
        return len(self.captions)


def train(model, dataloader, optimizer):
  for i, batch in enumerate(tqdm(dataloader)):
    batch[0] = batch[0].to(device)
    loss = model(batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 100 == 0:
      print(loss.item())

def main():
    dataset = Dataset(image_path, captions_path, transforms.Compose(
        [transforms.Resize((image_size, image_size)), transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])]))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    model = create_model()
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    for i in range(epochs):
        model.train()
        train(model, dataloader, optim)


