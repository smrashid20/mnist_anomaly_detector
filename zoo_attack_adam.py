import math

import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from CNN.resnet import ResNet18
from load_data import load_data
from PIL import Image

# import torch.backends.cudnn as cudnn


# load the mnist dataset (images are resized into 32 * 32)
training_set, test_set = load_data(data='mnist')

# define the model
model = ResNet18(dim=1)

# detect if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# load the learned model parameters
model.load_state_dict(torch.load('./model_weights/cpu_model.pth'))

model.to(device)
model.eval()

# todo note below is an example of getting the Z(X) vector in the ZOO paper

'''
z = model(image)

# if we consider just one image with size (1, 1, 32, 32)
# z.size() :   (1, 10)  10 elements are corresponding to classes

'''

step_sizes = [0.1]


def save_image(image, filename):
    adv_image = image.clone().cpu().detach().numpy()
    pil_image = Image.fromarray((adv_image[0][0] * 255).astype(np.uint8))
    pil_image.save('{}.jpg'.format(filename))


def f_x(network, image, t_0):
    z = torch.softmax(network(image), dim=1)
    log_z = torch.log(z).cpu().detach().numpy()
    log_z_other = log_z.copy()
    log_z_other[0][t_0] = -math.inf
    return max(log_z[0][t_0] - max(log_z_other[0]), 0)


def grad_and_hessian(network, image, coordinate, t_0):
    c_x = int(coordinate / image.shape[2])
    c_y = int(coordinate % image.shape[2])
    e_i = torch.zeros(image.shape).to(device)
    e_i[0][0][c_x][c_y] = torch.tensor(1.).to(device)
    h = torch.tensor(0.0001).to(device)
    return ((f_x(network, image + h * e_i, t_0) - f_x(network, image - h * e_i, t_0)) / (2 * 0.0001),
            (f_x(network, image + h * e_i, t_0) - 2 * f_x(network, image + h * e_i, t_0) + f_x(network, image - h * e_i,
                                                                                               t_0)) / (
                    2 * 0.0001) ** 2)


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


def zoo_attack(network, image, t_0, fw, step_size):
    '''

    #todo you are required to complete this part
    :param network: the model
    :param image: one image with size: (1, 1, 32, 32) type: torch.Tensor()
    :param t_0: real label
    :return: return a torch tensor (attack image) with size (1, 1, 32, 32)
    '''

    convergence = False
    max_iteration = 35000
    # step_size = 0.1
    curr_img = image.clone()
    M_value = np.zeros((image.size()[2], image.size()[3]))
    v_value = np.zeros((image.size()[2], image.size()[3]))
    T_value = np.zeros((image.size()[2], image.size()[3]), dtype=np.int64)

    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 10e-8

    initial_x = (image.shape[2] / 2)
    initial_y = (image.shape[2] / 2)
    neighborhood = 2

    while max_iteration > 0 and not convergence:
        print("Iteration {}".format(35000 - max_iteration))
        c_x = min(max(initial_x + random.randint(-neighborhood, neighborhood), 0), image.shape[2] - 1)
        c_y = min(max(initial_y + random.randint(-neighborhood, neighborhood), 0), image.shape[3] - 1)
        coordinate = c_x * image.shape[2] + c_y

        # coordinate = random.randint(0, image.shape[2] * image.shape[3] - 1)
        g_i, _ = grad_and_hessian(network, curr_img, coordinate, t_0)

        c_x = int(coordinate / image.shape[2])
        c_y = int(coordinate % image.shape[2])

        T_value[c_x][c_y] += 1
        M_value[c_x][c_y] = beta_1 * M_value[c_x][c_y] + (1 - beta_1) * g_i
        v_value[c_x][c_y] = beta_2 * v_value[c_x][c_y] + (1 - beta_2) * (g_i ** 2)
        m_i_prime = M_value[c_x][c_y] / (1 - beta_1**T_value[c_x][c_y])
        v_i_prime = v_value[c_x][c_y] / (1 - beta_2**T_value[c_x][c_y])
        delta_star = - step_size * (m_i_prime / math.sqrt(v_i_prime + epsilon))

        noise = torch.zeros(image.size()).to(device)

        # print(c_x, c_y)
        noise[0][0][c_x][c_y] = torch.tensor(delta_star).to(device)
        curr_img = curr_img + noise
        # print(curr_img[0][0][10])

        convergence = predict(network, curr_img, t_0)
        print("Current Confidence Score: {}".format(confidence(network, curr_img, t_0)))

        if max_iteration % 50 == 0:
            fw.write("Iteration {} Confidence {}\n"
                     .format(35000 - max_iteration, confidence(network, curr_img, t_0)))

        max_iteration -= 1

    fw.write("Total Iterations Done {}\n".format(35000 - max_iteration))
    return curr_img


# test the performance of attack
testloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2)


def get_target(labels):
    a = random.randint(0, 9)
    while a == labels[0]:
        a = random.randint(0, 9)
    return torch.tensor([a])


total = 0
success = 0
num_image = 10  # number of images to be attacked

for i, (images, labels) in enumerate(testloader):

    # plt.imshow(images[0][0])
    # plt.show()
    target_label = get_target(labels)
    images, labels = images.to(device), labels.to(device)
    # print(images, images.size())
    outputs = model(images)
    _, predicted = outputs.max(1)
    if predicted.item() != labels.item():
        continue

    save_image(images, 'original_images/image_{}'.format(i))

    total += 1

    # adv_image = zoo_attack(network=model, image=images, target=target_label)
    for step_size in step_sizes:
        with open('output_logs_adam/log_{}_{}.txt'.format(step_size, i), 'w') as fw:
            adv_image = zoo_attack(network=model, image=images, t_0=labels, fw=fw, step_size=step_size)
            adv_image = adv_image.to(device)
            adv_output = model(adv_image)
            _, adv_pred = adv_output.max(1)
            if adv_pred.item() != labels.item():
                fw.write("Original label {}, Predicted Label {}\n".format(labels.item(), adv_pred.item()))
                success += 1
                save_image(adv_image, 'output_images_adam/image_{}_{}'.format(step_size, i))

    if total >= num_image:
        break

print('success rate : %.4f' % (success / total))
