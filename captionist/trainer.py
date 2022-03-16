from tqdm import tqdm
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from data_loader import LoadData
from data_util import Collator
from model import EncoderDecoder
from utils import parameter_loader, save_model, show_image, view_image


def trainer(train):
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
    if train:
        for epoch in range(parameters['training_parameters']['num_epoch']):
            print(f'Epoch: {epoch + 1}')
            for idx, (images, captions) in tqdm(enumerate(iter(data_loader))):
                images, captions = images.to(device=device), captions.to(device=device)
                optimizer.zero_grad()
                outputs, attentions = model(images, captions)
                targets = captions[:, 1:]
                loss = criterion(outputs.view(-1, parameter_dict['vocab_size']), targets.reshape(-1))
                loss.backward()
                optimizer.step()
            save_model(model=model, parameters=parameters, epoch=epoch + 1)
    else:
        model.load_state_dict(torch.load('2022_03_08-12:24:48_PM.pth')['sate_dict'])
        model.eval()
        with torch.no_grad():
            input_img_org = cv2.imread('../data/test_image/dog.jpeg')
            input_img = cv2.resize(input_img_org, (256, 256), interpolation=cv2.INTER_LINEAR)
            input_img = torch.from_numpy(input_img)
            input_img = input_img.unsqueeze(0)
            input_img = input_img.permute(0, 3, 1, 2)
            input_img = input_img.float()
            encoded_image = model.image_encoder(input_img.to(device))
            caps, alphas = model.decoder.generate_caption(encoded_image, vocab=dataset.vocab)
            caption = ' '.join(caps)
            view_image(img=input_img_org, caption=caption)

