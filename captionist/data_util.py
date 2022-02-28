import torch
from torch.nn.utils.rnn import pad_sequence


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
