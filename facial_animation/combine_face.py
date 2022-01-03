import numpy as np
import os
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", metavar='DIR', help="your input folder location")
parser.add_argument("--output", metavar='DIR', help="output folder location")
args = parser.parse_args()

input_directory = args.input
save_directory = args.output

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

# seperate the image to left half and right half
def cut_image(img):
    im = Image.open(img)
    width, height = im.size
    left_half = (0, 0, width/2, height)
    right_half = (width/2, 0, width, height)
    im_l = im.crop(left_half)
    im_r = im.crop(right_half)
    return im_l, im_r

# Combine left half and right half to one image
def combine(input_directory, save_directory):
    input_d_l = os.listdir(input_directory)
    if '.DS_Store' in input_d_l:
        input_d_l.remove('.DS_Store')
    image_dir = sorted(input_d_l,  key = lambda x: int(x.split('_')[1]))
    for i in range(0, len(image_dir)):
        folder = sorted(os.listdir(os.path.join(input_directory, image_dir[i])))
        img_1 = os.path.join(input_directory, image_dir[i], folder[1])
        img_2 = os.path.join(input_directory, image_dir[i], folder[0])
        img_1_l, img_1_r = cut_image(img_1)
        img_2_l, img_2_r = cut_image(img_2)
        get_concat_h(img_1_l, img_2_r).save(os.path.join(save_directory, "img" + str(i+1).zfill(len(str(len(image_dir)))) + ".png"))


if os.path.exists(save_directory) == False:
    os.makedirs(save_directory)
combine(input_directory, save_directory)

