import yaml
import torch
from datetime import datetime
import matplotlib.pyplot as plt


def parameter_loader():
    with open('config.yml') as file:
        parameters = yaml.load(file, Loader=yaml.FullLoader)
    return parameters


def save_model(model, parameters, epoch):
    model_state = {
        'num_epoch': epoch,
        'embed_size': parameters['network_parameters']['embed_size'],
        'attention_dim': parameters['network_parameters']['attention_dim'],
        'encoder_dim': parameters['network_parameters']['encoder_dim'],
        'decoder_dim': parameters['network_parameters']['decoder_dim'],
        'state_dict': model.state_dict()
    }
    model_name = f'model_{epoch}.pth'
    torch.save(model_state, model_name)


def show_image(img, title=None):
    img[0] = img[0] * 0.229
    img[1] = img[1] * 0.224
    img[2] = img[2] * 0.225
    img[0] += 0.485
    img[1] += 0.456
    img[2] += 0.406
    img = img.numpy().transpose((1, 2, 0))
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.show()


def view_image(img, caption):
    plt.figure(figsize=(12, 7))
    plt.imshow(img)
    plt.title(caption)
    plt.xticks([])
    plt.yticks([])
    plt.show()