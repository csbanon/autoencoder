import torch
from torch import nn

class Autoencoder(nn.Module):

  def __init__(self):
    super(Autoencoder, self).__init__()

    self.encoder = nn.Sequential(
      nn.Conv2d(1, 16, 3, stride=1, padding=1),  # b, 16, 28, 28
      nn.ReLU(True),
      nn.MaxPool2d(3, stride=1, padding=1),  # b, 16, 28, 28
      nn.Conv2d(16, 8, 3, stride=1, padding=1),  # b, 8, 28, 28
      nn.ReLU(True),
      nn.MaxPool2d(3, stride=1, padding=1)  # b, 8, 28, 28
    )

    self.decoder = nn.Sequential(
      nn.Conv2d(8, 16, 3, stride=1, padding=1),  # b, 16, 28, 28
      nn.Upsample(size=[28, 28], mode='bilinear', align_corners=False),
      nn.Conv2d(16, 32, 3, stride=1, padding=1),  # b, 32, 28, 28
      nn.Upsample(size=[28, 28], mode='bilinear', align_corners=False),
      nn.Conv2d(32, 1, 3, stride=1, padding=1),  # b, 32, 28, 28
      nn.Tanh()
    )
    
  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x