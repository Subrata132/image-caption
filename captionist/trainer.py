import pandas as pd
import json
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from vocab_builder import Vocabulary
from data_util import data_loader_
from model import EncoderDecoder
from utils import save_model, view_image, plot_attention, load_train_test_val_data
from model_util import validator


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
    model = EncoderDecoder(
        parameter_dict=parameter_dict,
        device=device).to(device=device)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=parameters['learning_rate'])
    training_loss = []
    training_bleu = []
    validation_bleu = []
    results = {}
    if train:
        for epoch in range(parameters['num_epoch']):
            print(f'Epoch: {epoch + 1}')
            epoch_loss = 0
            counter = 0
            for idx, (images, captions) in enumerate(tqdm(iter(train_data_loader))):
                images, captions = images.to(device=device), captions.to(device=device)
                optimizer.zero_grad()
                outputs, attentions = model(images, captions)
                targets = captions[:, 1:]
                loss = criterion(outputs.view(-1, parameter_dict['vocab_size']), targets.reshape(-1))
                loss.backward()
                optimizer.step()
                epoch_loss = epoch_loss + loss.item()
                counter = counter + 1
            save_model(model=model, parameters=parameters, epoch=epoch + 1)
            training_loss.append(epoch_loss/counter)
            model.eval()
            with torch.no_grad():
                training_bleu.append(validator(model=model, data_loader=train_data_loader, vocab=vocab, device=device))
                validation_bleu.append(validator(model=model, data_loader=val_data_loader, vocab=vocab, device=device))
            model.train()
        results['loss'] = training_loss
        results['train_bleu'] = training_bleu
        results['val_bleu'] = validation_bleu
        with open('model_data/result_data.json', 'w') as f:
            json.dump(results, f)

    else:
        model.load_state_dict(torch.load(parameters['model_dir'])['state_dict'])
        model.eval()
        with torch.no_grad():
            input_img_org = Image.open('../data/test_image/dog_run.jpg').convert("RGB")
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

