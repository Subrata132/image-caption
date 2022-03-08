from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from data_loader import LoadData
from data_util import Collator
from model import EncoderDecoder
from utils import parameter_loader, save_model


def trainer():
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
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=parameters['training_parameters']['learning_rate'])
    for epoch in range(parameters['training_parameters']['num_epoch']):
        print(f'Epoch: {epoch+1}')
        for idx, (images, captions) in tqdm(enumerate(iter(data_loader))):
            images, captions = images.to(device=device), captions.to(device=device)
            optimizer.zero_grad()
            outputs, attentions = model(images, captions)
            targets = captions[:, 1:]
            loss = criterion(outputs.view(-1, parameter_dict['vocab_size']), targets.reshape(-1))
            loss.backward()
            optimizer.step()
        save_model(model=model, parameters=parameters, epoch=epoch+1)
