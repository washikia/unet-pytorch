from unet_pytorch.train import UNetTrainer
import os

inputs_dir = "D:\\washik_personal\\projects\\Unet\\unet-pytorch\\data\\inputs"
targets_dir = "D:\\washik_personal\\projects\\Unet\\unet-pytorch\\data\\targets"
model_dir = "D:\\washik_personal\\projects\\Unet\\unet-pytorch\\checkpoints"
os.makedirs(model_dir, exist_ok=True)

trainer = UNetTrainer(
    inputs_path=inputs_dir,
    targets_path=targets_dir,
    model_path=model_dir,
    batch_size=4
)
trainer.train(epochs=50, save_interval=10)