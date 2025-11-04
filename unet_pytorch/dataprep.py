'''
you get a dataset structured like this:
-processed
  -with_gates
    -2024
      -image_1.png
      -image_2.png
    -2025
      -image_3.png
      -image_4.png
  -without_gates
    -2024
      -image_1.png
      -image_2.png
    -2025
      -image_3.png
      -image_4.png

-label
    -label.json
      
and convert this to:

your_data_folder/
├── inputs/
│   ├── image1.tif
│   ├── image2.tif
│   └── ...
└── targets/
    ├── mask1.png
    ├── mask2.png
    └── ...

'''

from ntpath import isdir
import os
from glob import glob
from PIL import Image
import numpy as np
import json

def coord_to_png(coords: list[tuple[int, int]]) -> Image:
    width, height = 512, 256
    img = Image.new("L", (width, height), 0)
    for x, y in coords:
        img.putpixel((x, y), 255)
    return img


def make_dataset(input_images: str, input_labels: str, output_dir: str):
    # Create output directories if they do not exist
    os.makedirs(os.path.join(output_dir, "inputs"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "targets"), exist_ok=True)

    if not os.path.isdir(input_images):
        raise ValueError(f"Input images directory {input_images} does not exist")
    years = glob(os.path.join(input_images, "*"))

    with open(input_labels, "r") as f:
        label_data = json.load(f)

    image_num = 0
    for year in years:
      year = os.path.join(year, "without_gate")
      images = glob(os.path.join(year, "*.png"))
      print(f"Found {len(images)} images for {year}")
      for image in images:
          img = Image.open(image)
          image_name = os.path.basename(image)
          try:
            label_img = coord_to_png(label_data[image_name])
          except KeyError:
            label_img = None
          if label_img is not None:
            print(f"Label found for {image_name}")
            label_img.save(os.path.join(output_dir, "targets", f"mask_{image_num}.png"))
            img.save(os.path.join(output_dir, "inputs", f"image_{image_num}.tif"))
            image_num += 1
          else:
            print(f"No label found for {image_name}")
            continue



if __name__ == "__main__":
  make_dataset(input_images="D:\\washik_personal\\projects\\gate_prediction\\data\\processed", input_labels="D:\\washik_personal\\projects\\gate_prediction\\data\\labels\\annotations.json", output_dir="D:\\washik_personal\\projects\\Unet\\unet-pytorch\\data")