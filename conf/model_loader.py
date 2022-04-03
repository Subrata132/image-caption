import json
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from captionist.utils import load_train_test_val_data
from captionist.vocab_builder import Vocabulary
from captionist.data_util import data_loader_
from captionist.model import EncoderDecoder


def save_attention(img_name, result, attention_plot):
    temp_image = Image.open(f'uploaded_images/{img_name}').convert("RGB")
    fig = plt.figure(figsize=(10, 10))
    len_result = len(result)
    for l in range(len_result):
        temp_att = attention_plot[l].reshape(8, 8)
        ax = fig.add_subplot(len_result // 2, len_result // 2, l + 1)
        ax.set_title(result[l], fontsize=12)
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.7, extent=img.get_extent())
        ax.set_xticks([], minor=False)
        ax.set_yticks([], minor=False)
    plt.tight_layout()
    plt.savefig(f'attention_images/attention_{img_name.split(".")[0]}.png', dpi=150)


def load_model(parameters):
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
    train_data_loader, vocab = data_loader_(df=train_df, parameters=parameters, vocabulary=vocabulary,
                                            transform=transform)
    val_data_loader, _ = data_loader_(df=val_df, parameters=parameters, vocabulary=vocabulary, transform=transform)
    test_data_loader, _ = data_loader_(df=test_df, parameters=parameters, vocabulary=vocabulary,
                                       transform=transform)
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
    model.load_state_dict(torch.load(parameters['model_dir'])['state_dict'])
    model.eval()
    return model, device, vocab


class Explainer:
    def __init__(self):
        with open('model_data/parameters.json') as file:
            self.parameters = json.load(file)
        self.model, self.device, self.vocab = load_model(self.parameters)

    def caption_generator(self, image_name):
        image = Image.open(f'uploaded_images/{image_name}').convert("RGB")
        with torch.no_grad():
            transform_ = T.Compose([
                T.Resize((256, 256)),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            input_img = transform_(image)
            input_img = input_img.unsqueeze(0)
            encoded_image = self.model.image_encoder(input_img.to(self.device))
            caps, alphas = self.model.decoder.generate_caption(encoded_image, vocab=self.vocab)
            save_attention(image_name, caps, alphas)
            data = {
                'caption': ' '.join(caps[:-1]),
                'att_url': f'attention_images/attention_{image_name.split(".")[0]}.png'
            }
            return data
