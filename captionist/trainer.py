from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from data_loader import LoadData
from data_util import Collator
from model import EncoderDecoder
from utils import parameter_loader, save_model, show_image


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
        print_every = 100
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
                if (idx + 1) % print_every == 0:
                    print("Epoch: {} loss: {:.5f}".format(epoch, loss.item()))
                    model.eval()
                    with torch.no_grad():
                        dataiter = iter(data_loader)
                        img, _ = next(dataiter)
                        features = model.image_encoder(img[0:1].to(device))
                        caps, alphas = model.decoder.generate_caption(features, vocab=dataset.vocab)
                        caption = ' '.join(caps)
                        show_image(img[0], title=caption)
                    model.train()
            save_model(model=model, parameters=parameters, epoch=epoch+1)
    else:
        model.load_state_dict(torch.load('model_data/attention_model_state.pth')['state_dict'])
        print('model loaded')
