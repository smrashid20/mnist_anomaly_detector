import math
import os.path

import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from CNN.resnet import ResNet18
from load_data import load_data
from PIL import Image

training_set, test_set = load_data(data='mnist')
testloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2)


def delete_contents(directory):
    for root, dirs, files in os.walk(directory, topdown=False):
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            os.rmdir(dir_path)

def save_image(image, filename):
    pil_image = Image.fromarray((image[0][0] * 255).astype(np.uint8))
    pil_image.save('{}.jpg'.format(filename))


if not os.path.exists('clean_digits'):
    os.mkdir('clean_digits')
else:
    delete_contents('clean_digits')

for i in range(10):
    if not os.path.exists('clean_digits/' + str(i)):
        os.mkdir('clean_digits/' + str(i))

digit_count = [0] * 10
print(len(testloader))
for i, (images, labels) in enumerate(testloader):
    if sum(digit_count) >= 10 * 20:
        break
    digit = int(labels.cpu().detach().numpy()[0])
    if digit_count[digit] < 20:
        digit_count[digit] += 1
        save_image(images.numpy(), os.path.join('clean_digits/' + str(digit), 'image_{}_{}'.
                                            format(digit,digit_count[digit])))

