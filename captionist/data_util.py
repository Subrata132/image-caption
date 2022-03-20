import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from data_loader import LoadData


class Collator:
    def __init__(self, pad_idx, batch_first=False):
        self.pad_idx = pad_idx
        self.batch_first = batch_first

    def __call__(self, batch):
        images = []
        targets = []
        for item in batch:
            image = item[0]
            targets.append(item[1])
            images.append(image.unsqueeze(0))
        images = torch.cat(images, dim=0)
        targets = pad_sequence(targets, batch_first=self.batch_first, padding_value=self.pad_idx)
        return images, targets


def data_loader_(df, parameters, vocabulary, transform):
    dataset = LoadData(
        image_dir=parameters['img_dir'],
        caption_df=df,
        vocab=vocabulary,
        transform=transform
    )
    pad_idx = dataset.vocab.stoi['<PAD>']
    collator = Collator(
        pad_idx=pad_idx,
        batch_first=True
    )
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=parameters['batch_size'],
        num_workers=parameters['num_workers'],
        shuffle=True,
        collate_fn=collator
    )
    return data_loader, dataset.vocab
