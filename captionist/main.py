import argparse
from trainer import trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=False, action="store_true")
    args = parser.parse_args()
    train = args.train
    trainer(train)


if __name__ == '__main__':
    main()
