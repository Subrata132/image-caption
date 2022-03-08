import yaml


def parameter_loader():
    with open('config.yml') as file:
        parameters = yaml.load(file, Loader=yaml.FullLoader)
    return parameters
