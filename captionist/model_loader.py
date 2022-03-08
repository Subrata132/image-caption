from model import EncoderDecoder
from utils import parameter_loader


def load_trained_model():
    parameters = parameter_loader()
    model = EncoderDecoder()