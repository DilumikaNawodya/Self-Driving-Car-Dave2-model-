import os
import numpy as np
import pandas as pd
import cv2
from scipy.ndimage import rotate
from scipy.stats import bernoulli
import matplotlib.pyplot as plt

IMAGE_PATH = 'image data path goes here'
CSV_PATH = 'csv file path goes here'

def crop(image):
    
    top = int(np.ceil(image.shape[0] * 0.35))
    bottom = image.shape[0] - int(np.ceil(image.shape[0] * 0.10))
    
    return image[top:bottom, :]

def resize(image):
    
    return cv2.resize(image, (64, 64))

def random_shear(image, str_angle):
    
    rows, cols, ch = image.shape
    
    l = np.random.randint(-200, 201)
    rpt = [cols/2+l, rows/2]
    mt1 = np.float32(
        [[0, rows],
        [cols, rows],
        [cols/2, rows/2]]
    )
    mt2 = np.float32(
        [[0, rows],
        [cols, rows],
        rpt]
    )
    
    dstr = l / (rows/2) * 360 / (2*np.pi*25.0) / 6.0
    M = cv2.getAffineTransform(mt1, mt2)
    image = cv2.warpAffine(image, M, (cols, rows), borderMode=1)
    str_angle += dstr
    
    return image, str_angle

def random_flip(image, str_angle):
    
    return np.fliplr(image), -1*str_angle

def random_gamma(image):
    
    gamma = np.random.uniform(0.4, 1.5)
    inv = 1.0 / gamma
    
    table = np.array([(
        (i/255.0) ** inv) * 255 for i in np.arange(0, 256)
    ]).astype("uint8")
    
    return cv2.LUT(image, table)

def random_rotation(image, str_angle):
    
    angle = np.random.uniform(-15, 16)
    rad = (np.pi/180.0) * angle
    
    return rotate(image, angle, reshape=False), str_angle + (-1)*rad

def augment(image, str_angle):
    
    head = bernoulli.rvs(0.9)
    
    if head:
        image, str_angle = random_shear(image, str_angle)
    
    image = crop(image)
    image, str_angle = random_flip(image, str_angle)
    image = random_gamma(image)
    image = resize(image)
    
    return image, str_angle

def get_next_image_files():
    
    data = pd.read_csv(CSV_PATH)
    num_of_img = len(data)
    rnd_indices = np.random.randint(0, num_of_img, 64)

    image_files_and_angles = []
    for index in rnd_indices:
        rnd_image = np.random.randint(0, 3)
        if rnd_image == 0:
            img = data.iloc[index]['left'].strip()
            angle = data.iloc[index]['steering'] + 0.229
            image_files_and_angles.append((img, angle))

        elif rnd_image == 1:
            img = data.iloc[index]['center'].strip()
            angle = data.iloc[index]['steering']
            image_files_and_angles.append((img, angle))
        else:
            img = data.iloc[index]['right'].strip()
            angle = data.iloc[index]['steering'] - 0.229
            image_files_and_angles.append((img, angle))

    return image_files_and_angles


def generate_batch():
    
    while True:
        X = []
        y = []
        images = get_next_image_files()
        for img_file, angle in images:
            raw_image = plt.imread(IMAGE_PATH + img_file)
            raw_angle = angle
            new_image, new_angle = augment(raw_image, raw_angle)
            X.append(new_image/127.5-1.0)
            y.append(new_angle)

        yield np.array(X), np.array(y)