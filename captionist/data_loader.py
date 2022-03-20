import pandas as pd
from PIL import Image
from vocab_builder import Vocabulary
import torch


class LoadData:
    def __init__(self, image_dir, caption_df, vocab, transform=None, freq_threshold=5):
        self.image_dir = image_dir
        self.caption_df = caption_df
        self.image_names = self.caption_df['image']
        self.captions = self.caption_df['caption']
        self.vocab = vocab
        self.transform = transform

    def __len__(self):
        return self.caption_df.shape[0]

    def __getitem__(self, idx):
        caption = self.captions[idx]
        img_name = self.image_names[idx]
        img = Image.open(self.image_dir + img_name).convert("RGB")
        if self.transform:
            img = self.transform(img)
        caption_vector = []
        caption_vector += [self.vocab.stoi['<SOS>']]
        caption_vector += self.vocab.string_to_numerical(caption)
        caption_vector += [self.vocab.stoi['<EOS>']]
        return img, torch.tensor(caption_vector)
