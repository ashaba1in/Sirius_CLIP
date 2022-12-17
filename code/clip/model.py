import torchvision
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = 'resnet50'
repo = "pytorch/vision"
weights = "IMAGENET1K_V2"
bert_model_name = 'bert-base-uncased'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextEncoder(nn.Module):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        for p in self.model.parameters():
            p.require_grad = True
        self.ind = 0

    def forward(self, x):
        # print(x)
        x = self.tokenizer(x, padding=True, return_tensors='pt')
        for key in x:
            x[key] = x[key].to(device)
        # print(x)
        output = self.model(**x)
        # output = output.to(device)
        return output.last_hidden_state[:, self.ind, :]


class Projection(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = x / x.norm(dim=1, keepdim=True)
        return x


class CLIP(nn.Module):
    def __init__(self, image_encoder, text_encoder, input_image, input_text, out_size=256):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.projection_text = Projection(input_text, out_size)
        self.projection_image = Projection(input_image, out_size)
        self.temperature = 1

    def forward(self, batch):
        img = batch[0]
        # print(img.dtype, img.shape)
        text = batch[1]
        img = self.image_encoder(img)
        text = self.text_encoder(text)

        img = self.projection_image(img)
        # print(text.shape)
        text = self.projection_text(text)

        dist = img @ text.T
        # li = F.softmax(dist, dim=1)
        # ti = F.softmax(dist, dim=0)
        ce = nn.CrossEntropyLoss()
        loss_i = ce(dist, torch.arange(len(batch[0])).to(device))
        loss_t = ce(dist.T, torch.arange(len(batch[0])).to(device))
        return (loss_i + loss_t) / 2

def create_model():
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    text_encoder = BertModel.from_pretrained(bert_model_name)
    image_encoder = torch.hub.load(repo, model_name, weights=weights)
    image_encoder.fc = nn.Linear(2048, 512)
    text_encoder_model = TextEncoder(text_encoder, tokenizer)
    model = CLIP(image_encoder, text_encoder_model, 512, 768)
    model = model.to(device)
    return model
