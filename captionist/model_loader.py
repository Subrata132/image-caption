from model import EncoderDecoder
from utils import parameter_loader
from data_loader import LoadData


def load_trained_model():
    parameters = parameter_loader()
    model = EncoderDecoder()

