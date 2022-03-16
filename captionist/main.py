import argparse
# from trainer import trainer
from trainer_kaggle import trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=False, action="store_true")
    parser.add_argument("--attention_dim", default=256, type=int)
    parser.add_argument("--encoder_dim", default=2048, type=int)
    parser.add_argument("--decoder_dim", default=512, type=int)
    parser.add_argument("--embed_size", default=300, type=int)
    parser.add_argument("--drop_rate", default=0.3, type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--num_epoch", type=int)
    parser.add_argument("--learning_rate", default=0.0003, type=float)
    parser.add_argument("--img_dir",  type=str)
    parser.add_argument("--caption_dir", type=str)
    parser.add_argument("--model_dir", default='', type=str)
    args = parser.parse_args()
    train = args.train
    parameters = {
        'attention_dim': args.attention_dim,
        'encoder_dim': args.encoder_dim,
        'decoder_dim': args.decoder_dim,
        'embed_size': args.embed_size,
        'drop_rate': args.drop_rate,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'num_epoch': args.num_epoch,
        'learning_rate': args.learning_rate,
        'img_dir': args.img_dir,
        'caption_dir': args.caption_dir,
        'model_dir': args.model_dir
    }
    trainer(train, parameters=parameters)


if __name__ == '__main__':
    main()
