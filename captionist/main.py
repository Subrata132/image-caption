import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from data_loader import LoadData
from data_util import Collator
from model import EncoderDecoder
from utils import parameter_loader


def main():
    parameters = parameter_loader()
    batch_size = parameters['training_parameters']['batch_size']
    num_worker = parameters['training_parameters']['num_workers']
    transform = T.Compose([
        T.Resize(256),
        T.RandomCrop(224),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    dataset = LoadData(
        image_dir=parameters['data_locations']['img_dir'],
        caption_dir=parameters['data_locations']['caption_dir'],
        transform=transform
    )
    pad_idx = dataset.vocab.stoi['<PAD>']
    collator = Collator(
        pad_idx=pad_idx,
        batch_first=True
    )
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_worker,
        shuffle=True,
        collate_fn=collator
    )
    parameter_dict = parameters['network_parameters']
    parameter_dict['vocab_size'] = len(dataset.vocab)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EncoderDecoder(
        parameter_dict=parameter_dict,
        device=device).to(device=device)
    images, captions = next(iter(data_loader))
    predictions, alphas = model(images.to(device), captions.to(device))
    print(images.shape, captions.shape, predictions.shape, alphas.shape)
    images, captions = next(iter(data_loader))
    predictions, alphas = model(images.to(device), captions.to(device))
    print(images.shape, captions.shape, predictions.shape, alphas.shape)


if __name__ == '__main__':
    main()
