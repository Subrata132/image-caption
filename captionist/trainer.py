import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from vocab_builder import Vocabulary
from data_loader import LoadData
from data_util import Collator, data_loader_
from model import EncoderDecoder
from utils import save_model, view_image, plot_attention, load_train_test_val_data, show_image


def trainer(train, parameters):
    all_data_df = pd.read_csv(parameters['caption_dir'])
    train_df, val_df, test_df = load_train_test_val_data(all_data_df=all_data_df)
    vocabulary = Vocabulary()
    vocabulary.build_vocab(all_data_df['caption'].tolist())
    transform = T.Compose([
        T.Resize(256),
        T.RandomCrop(224),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    train_data_loader, vocab = data_loader_(df=train_df, parameters=parameters, vocabulary=vocabulary, transform=transform)
    val_data_loader, _ = data_loader_(df=val_df, parameters=parameters, vocabulary=vocabulary, transform=transform)
    test_data_loader, _ = data_loader_(df=test_df, parameters=parameters, vocabulary=vocabulary, transform=transform)
    parameter_dict = {
        'attention_dim': parameters['attention_dim'],
        'encoder_dim': parameters['encoder_dim'],
        'decoder_dim': parameters['decoder_dim'],
        'embed_size': parameters['embed_size'],
        'drop_rate': parameters['drop_rate'],
        'vocab_size': len(vocab)
    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = EncoderDecoder(
        parameter_dict=parameter_dict,
        device=device).to(device=device)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=parameters['learning_rate'])
    if train:
        print('ok')
        for epoch in range(parameters['num_epoch']):
            print(f'Epoch: {epoch + 1}')
            for idx, (images, captions) in tqdm(enumerate(iter(train_data_loader))):
                images, captions = images.to(device=device), captions.to(device=device)
                optimizer.zero_grad()
                outputs, attentions = model(images, captions)
                targets = captions[:, 1:]
                loss = criterion(outputs.view(-1, parameter_dict['vocab_size']), targets.reshape(-1))
                loss.backward()
                optimizer.step()
                if (idx + 1) % 100 == 0:
                    print("Epoch: {} loss: {:.5f}".format(epoch, loss.item()))
                    model.eval()
                    with torch.no_grad():
                        dataiter = iter(train_data_loader)
                        img, _ = next(dataiter)
                        features = model.image_encoder(img[0:1].to(device))
                        caps, alphas = model.decoder.generate_caption(features, vocab=vocab)
                        caption = ' '.join(caps)
                        print(caption)
                        show_image(img[0], title=caption)
                    model.train()
            save_model(model=model, parameters=parameters, epoch=epoch + 1)
    else:
        model.load_state_dict(torch.load(parameters['model_dir'])['state_dict'])
        model.eval()
        with torch.no_grad():
            input_img_org = Image.open('../data/test_image/dog.jpeg').convert("RGB")
            transform_ = T.Compose([
                T.Resize((256, 256)),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            input_img = transform_(input_img_org)
            input_img = input_img.unsqueeze(0)
            encoded_image = model.image_encoder(input_img.to(device))
            caps, alphas = model.decoder.generate_caption(encoded_image, vocab=vocab)
            caption = ' '.join(caps[:-1])
            view_image(img=input_img_org, caption=caption)
            plot_attention(input_img_org, caps, alphas)
            plt.show()

