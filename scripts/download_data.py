from geode.datamodules import MNISTDataModule, OmniglotDataModule


if __name__ == '__main__':
    omniglot_datamodule = OmniglotDataModule(
        data_path='./data',
        batch_size=64,
        num_workers=4,
    )
    omniglot_datamodule.setup(download=True)

    mnist_datamodule = MNISTDataModule(
        data_path='./data',
        batch_size=64,
        num_workers=4,
    )
    mnist_datamodule.setup(download=True)

