import math
import os.path
import random

import numpy as np
import torch
from PIL import Image

from CNN.resnet import ResNet18

# define the model
model = ResNet18(dim=1)

# detect if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# load the learned model parameters
model.load_state_dict(torch.load('./model_weights/cpu_model.pth'))

model.to(device)
model.eval()

STEP_SIZE = 0.5
TARGET_DIGITS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
ATTACK_PROBABILITY = 0.5
random.seed(1)


def random_binary(x):
    if random.random() <= x:
        return 1
    else:
        return 0


def delete_contents(directory):
    for root, dirs, files in os.walk(directory, topdown=False):
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            os.rmdir(dir_path)


def save_image(image, filename):
    pil_image = Image.fromarray((image * 255).astype(np.uint8))
    pil_image.save('{}'.format(filename))


def f_x(network, image, t_0):
    network = network.float()
    z = torch.softmax(network(image.float()), dim=1)
    log_z = torch.log(z).cpu().detach().numpy()
    log_z_other = log_z.copy()
    log_z_other[0][t_0] = -math.inf
    return max(log_z[0][t_0] - max(log_z_other[0]), 0)


def grad_and_hessian(network, image, coordinate, t_0):
    c_x = int(coordinate / image.shape[2])
    c_y = int(coordinate % image.shape[2])
    e_i = torch.zeros(image.shape).to(device)
    e_i[0][0][c_x][c_y] = torch.tensor(1.).to(device)
    h = torch.tensor(0.001).to(device)
    return ((f_x(network, image + h * e_i, t_0) - f_x(network, image - h * e_i, t_0)) / (2 * 0.001),
            (f_x(network, image + h * e_i, t_0) - 2 * f_x(network, image + h * e_i, t_0) + f_x(network, image - h * e_i,
                                                                                               t_0)) / (
                    2 * 0.001) ** 2)


def predict(network, image, t_0):
    z = torch.softmax(network(image), dim=1)
    pred = torch.argmax(z, dim=1)
    # print(z)
    # print(pred, t_0)
    if pred != t_0:
        return True
    return False


def confidence(network, image, t_0):
    z = torch.softmax(network(image), dim=1).cpu().detach().numpy()
    z_other = z.copy()
    z_other[0][t_0] = 0
    return np.max(z) - np.max(z_other)


def zoo_attack(network, image, t_0, step_size_):
    convergence = False
    max_iteration = 5000
    curr_img = image.clone()

    initial_x = (image.shape[2] / 2)
    initial_y = (image.shape[2] / 2)
    neighborhood = 2

    while max_iteration > 0 and not convergence:

        c_x = min(max(initial_x + random.randint(-neighborhood, neighborhood), 0), image.shape[2] - 1)
        c_y = min(max(initial_y + random.randint(-neighborhood, neighborhood), 0), image.shape[3] - 1)
        coordinate = c_x * image.shape[2] + c_y

        #coordinate = random.randint(0, image.shape[2] * image.shape[3] - 1)
        g_i, h_i = grad_and_hessian(network, curr_img, coordinate, t_0)
        if h_i <= 0:
            delta_star = -step_size_ * g_i
        else:
            delta_star = -step_size_ * (g_i / h_i)

        noise = torch.zeros(image.size()).to(device)
        c_x = int(coordinate / image.shape[2])
        c_y = int(coordinate % image.shape[2])
        # print(c_x, c_y)
        noise[0][0][c_x][c_y] = torch.tensor(delta_star).to(device)
        curr_img = curr_img + noise
        # print(curr_img[0][0][10])

        convergence = predict(network, curr_img, t_0)

        if (5000 - max_iteration) % 25 == 0:
            print("Iteration {}".format(5000 - max_iteration))
            print("Current Confidence Score: {}".format(confidence(network, curr_img, t_0)))

        max_iteration -= 1

    return curr_img.cpu().detach().numpy()[0][0]


if not os.path.exists('adversarial_digits_{}'.format(STEP_SIZE)):
    os.mkdir('adversarial_digits_{}'.format(STEP_SIZE))
else:
    delete_contents('adversarial_digits_{}'.format(STEP_SIZE))

for i in range(10):
    if not os.path.exists('adversarial_digits_{}'.format(STEP_SIZE) + "/" + str(i)):
        os.mkdir('adversarial_digits_{}'.format(STEP_SIZE) + "/" + str(i))

if not os.path.exists('adversarial_labels_{}'.format(STEP_SIZE)):
    os.mkdir('adversarial_labels_{}'.format(STEP_SIZE))
else:
    delete_contents('adversarial_labels_{}'.format(STEP_SIZE))

adv_labels_all_digits = dict()

for target_digit in TARGET_DIGITS:
    adv_label_digit = list()
    files = os.listdir(os.path.join('clean_digits', str(target_digit)))
    for file in sorted(files):
        print("File {}".format(file))
        image = Image.open(os.path.join(os.path.join('clean_digits', str(target_digit)), file))
        np_img_arr = np.array(image, dtype=np.float64) / 255
        reshaped_np_img = np.reshape(np_img_arr, (1, 1, np_img_arr.shape[0], np_img_arr.shape[1]))

        target_label = torch.tensor([target_digit]).to(device)
        target_torch_img = torch.tensor(reshaped_np_img).to(device)

        if random_binary(ATTACK_PROBABILITY) == 1:
            print("Creating adv.")
            adv_img = zoo_attack(model.float(), target_torch_img.float(), target_label, STEP_SIZE)
            save_image(adv_img, os.path.join('adversarial_digits_{}'.format(STEP_SIZE) + "/" + str(target_digit), file))
            adv_label_digit.append([file, 1])
        else:
            print("Placing as is.")
            save_image(np_img_arr, os.path.join('adversarial_digits_{}'.format(STEP_SIZE) + "/" + str(target_digit),
                                                file))
            adv_label_digit.append([file, 0])
    adv_labels_all_digits[target_digit] = adv_label_digit

for k, v in adv_labels_all_digits.items():
    with open(os.path.join('adversarial_labels_{}'.format(STEP_SIZE), str(k) + '.txt'), 'w') as f:
        for item in v:
            f.write("{} {}\n".format(item[0], item[1]))
