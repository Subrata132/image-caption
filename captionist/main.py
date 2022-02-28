from torch.utils.data import DataLoader
import torchvision.transforms as T
from data_loader import LoadData
from data_util import Collator


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
    images, captions = next(iter(data_loader))
    print(images.shape, captions.shape)
    images, captions = next(iter(data_loader))
    print(images.shape, captions.shape)


if __name__ == '__main__':
    main()
