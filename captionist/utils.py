import yaml
import torch
from datetime import datetime


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
        'sate_dict': model.state_dict()
    }
    model_name = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p") + '.pth'
    torch.save(model_state, model_name)
