import numpy as np
import os
from PIL import Image


def get_img(i, data_root, rgb=False):
    image_list = os.listdir(data_root)
    # image_name = f"image{i:02d}.jpg"
    if not rgb:
        image_name = list(filter(lambda x: f"image{i:02d}_" in x, image_list))[0]
        rgb2gray = lambda img_rgb: img_rgb[:, :, 0] * .299 + img_rgb[:, :, 1] * .587 + img_rgb[:, :, 2] * .114
        myImage = Image.open(data_root + image_name)
        img_rgb = np.array(myImage)
        img_gray = rgb2gray(img_rgb)
        return img_gray
    else:
        image_name = list(filter(lambda x: f"image_{i:02d}." in x, image_list))[0]
        myImage = Image.open(data_root + image_name)
        img_rgb = np.array(myImage)
        return img_rgb


def get_imgs(n, data_root):
    image_list = os.listdir(data_root)
    # First read a sample image to obtain its height and width:
    image_name = list(filter(lambda x: f"image{1:02d}_" in x, image_list))[0]
    rgb2gray = lambda img_rgb: img_rgb[:, :, 0] * .299 + img_rgb[:, :, 1] * .587 + img_rgb[:, :, 2] * .114
    myImage = Image.open(data_root + image_name)
    img_rgb = np.array(myImage)
    img_gray = rgb2gray(img_rgb)
    img_height, img_width = img_gray.shape

    imgs = np.zeros(shape=(n, img_height, img_width))
    image_list = os.listdir(data_root)
    for i in range(n):
        image_name = list(filter(lambda x: f"image{i + 1:02d}_" in x, image_list))[0]
        myImage = Image.open(data_root + image_name)
        img_rgb = np.array(myImage)
        img_gray = rgb2gray(img_rgb)
        imgs[i, :, :] = img_gray
    return imgs

def get_euler_data(data_root, filename="eul_data", n=995):
    euler_data = np.zeros(shape=(n, 3))
    with open(data_root + "/" + filename + '.txt', 'r') as f:
        for i, line in enumerate(f):
            corrected_line = line.replace("(", "").replace(")", "").replace(" ", "").replace("\n", "").split(',')
            if "None" in corrected_line:
                corrected_line = previous_line
            euler_data[i, :] = np.array([float(x) for x in corrected_line])
            previous_line = corrected_line
    return euler_data


def get_quat_data(data_root, filename="quat_data", n=995):
    quat_data = np.zeros(shape=(n, 4))
    with open(data_root + "/" + filename + '.txt', 'r') as f:
        for i, line in enumerate(f):
            corrected_line = line.replace("(", "").replace(")", "").replace(" ", "").replace("\n", "").split(',')
            if "None" in corrected_line:
                corrected_line = previous_line
            quat_data[i, :] = np.array([float(x) for x in corrected_line])
            previous_line = corrected_line
    return quat_data
