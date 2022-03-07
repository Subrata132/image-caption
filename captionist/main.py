import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from data_loader import LoadData
from data_util import Collator
from model import ImageEncoder


def main():
    batch_size = 32
    num_worker = 0
    transform = T.Compose([
        T.Resize(256),
        T.RandomCrop(224),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    dataset = LoadData(
        image_dir='../data/Images/',
        caption_dir='../data/captions.txt',
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image_encoder = ImageEncoder().to(device=device)
    images, captions = next(iter(data_loader))
    image_encoded = image_encoder(images.to(device))
    print(images.shape, captions.shape, image_encoded.shape, image_encoded[0].shape)
    images, captions = next(iter(data_loader))
    image_encoded = image_encoder(images.to(device))
    print(images.shape, captions.shape, image_encoded.shape, image_encoded[0].shape)


if __name__ == '__main__':
    main()
