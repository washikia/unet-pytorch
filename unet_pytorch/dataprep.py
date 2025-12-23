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
from PIL import Image, ImageDraw
import numpy as np
import json


def coord_to_png(coords: list[tuple[int, int]], radius: int = 3) -> Image:
    width, height = 512, 256
    img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(img)

    for x, y in coords:
        left_up = (x - radius, y - radius)
        right_down = (x + radius, y + radius)
        draw.ellipse([left_up, right_down], fill=255)

    return img



# the function receives the input directory containing all the images
# and the path to the json file
# it takes the names of the images, and checks for that image name in the json file.
# Once it has found the file, it creates another images and puts it in the target folder

def make_dataset_2(input_images: str, input_labels: str):
    # check if the image directory is valid
    if not os.path.isdir(input_images):
       raise ValueError(f"Input images directory {input_images} does not exist")

    # Create output directory
    parent_dir = os.path.dirname(input_images)
    output_dir = os.path.join(parent_dir, "targets")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(input_labels, "r") as f:
       label_data = json.load(f)
    
    images = glob(os.path.join(input_images, "*.png"))
    for image_path in images:
      img_name = os.path.basename(image_path)
      label = label_data.get(img_name)
      if label is None:
         continue
      target_img = coord_to_png(label, 3)
      target_img.save(os.path.join(output_dir, img_name))
      


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