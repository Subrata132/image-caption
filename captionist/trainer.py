from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from data_loader import LoadData
from data_util import Collator
from model import EncoderDecoder
from utils import save_model, view_image, plot_attention


def trainer(train, parameters):
    batch_size = parameters['batch_size']
    num_worker = parameters['num_workers']
    transform = T.Compose([
        T.Resize(256),
        T.RandomCrop(224),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    dataset = LoadData(
        image_dir=parameters['img_dir'],
        caption_dir=parameters['caption_dir'],
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
    parameter_dict = {
        'attention_dim': parameters['attention_dim'],
        'encoder_dim': parameters['encoder_dim'],
        'decoder_dim': parameters['decoder_dim'],
        'embed_size': parameters['embed_size'],
        'drop_rate': parameters['drop_rate'],
        'vocab_size': len(dataset.vocab)
    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EncoderDecoder(
        parameter_dict=parameter_dict,
        device=device).to(device=device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=parameters['learning_rate'])
    if train:
        for epoch in range(parameters['num_epoch']):
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
        model.load_state_dict(torch.load(parameters['model_dir'])['state_dict'])
        model.eval()
        with torch.no_grad():
            input_img_org = Image.open('../data/test_image/test.jpg').convert("RGB")
            transform_ = T.Compose([
                T.Resize((256, 256)),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            input_img = transform_(input_img_org)
            input_img = input_img.unsqueeze(0)
            encoded_image = model.image_encoder(input_img.to(device))
            caps, alphas = model.decoder.generate_caption(encoded_image, vocab=dataset.vocab)
            caption = ' '.join(caps[:-1])
            view_image(img=input_img_org, caption=caption)
            plot_attention(input_img_org, caps, alphas)
            plt.show()

