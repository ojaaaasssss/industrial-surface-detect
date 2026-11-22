import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from PIL import Image

LABELS = ['non_defective', 'defective']
SIZE = 128

def transform(training=False):
    ops = [transforms.Resize((SIZE, SIZE))]
    if training:
        ops += [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.RandomRotation(10)]
    ops += [transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)]
    return transforms.Compose(ops)

class DefectCNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers=[]; inc=3
        for out in (16,32,64,128):
            layers += [nn.Conv2d(inc,out,3,padding=1),nn.BatchNorm2d(out),nn.ReLU(inplace=True),nn.MaxPool2d(2)]
            inc=out
        self.features=nn.Sequential(*layers)
        self.head=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Dropout(0.25),nn.Linear(128,1))
    def forward(self,x):
        return self.head(self.features(x)).squeeze(1)

def load_model(path, device='cpu'):
    # Only load trusted checkpoint files.
    checkpoint=torch.load(path,map_location=device,weights_only=True)
    model=DefectCNN().to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, checkpoint

def predict(model, image, device='cpu', threshold=0.5):
    image=image.convert('RGB')
    x=transform()(image).unsqueeze(0).to(device)
    with torch.no_grad():
        prob=torch.sigmoid(model(x)).item()
    return {'label':LABELS[int(prob>=threshold)],'defect_probability':round(prob,4),'confidence':round(max(prob,1-prob),4)}
