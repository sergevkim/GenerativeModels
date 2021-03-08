from geode.datamodules import (
    CelebADataModule,
    MNISTDataModule,
    OmniglotDataModule,
)


from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)


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

    celeba_datamodule = CelebADataModule(
        data_path='./data/celeba',
        batch_size=64,
        num_workers=4,
    )
    celeba_datamodule.setup(download=True)

