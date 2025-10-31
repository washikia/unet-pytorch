import torch

from unet_pytorch.model import UNet

model = UNet(in_channels=1, out_channels=2)

input_tensor = torch.randn(1, 1, 512, 512)

output_tensor = model(input_tensor)

print(output_tensor.shape)  # Should be (1, 2, 512, 512)