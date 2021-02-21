from argparse import ArgumentParser
from pathlib import Path

from geode.datamodules import OmniglotDataModule
from geode.loggers import NeptuneLogger
from geode.models import SimpleAutoencoder
from geode.trainer import Trainer
from geode.utils.randomer import Randomer

from configs.autoencoder_config import (
    CommonArguments,
    DataArguments,
    TrainArguments,
    SpecificArguments,
)


def main(args):
    Randomer.set_seed(seed=args.seed)

    model = SimpleAutoencoder(
        hidden_dim=args.hidden_dim,
        n_blocks=args.n_blocks,
        device=args.device,
        learning_rate=args.learning_rate,
    ).to(args.device)

    datamodule = OmniglotDataModule(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    datamodule.setup(val_ratio=args.val_ratio)

    logger = NeptuneLogger(
        api_token=args.neptune_api_token,
        project_name=args.neptune_project_name,
        experiment_name=args.neptune_experiment_name,
    )

    trainer = Trainer(
        logger=logger,
        max_epoch=args.max_epoch,
        verbose=args.verbose,
        version=args.version,
    )
    trainer.fit(
        model=model,
        datamodule=datamodule,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    default_args_dict = {
        **vars(CommonArguments()),
        **vars(DataArguments()),
        **vars(TrainArguments()),
        **vars(SpecificArguments()),
    }

    for arg, value in default_args_dict.items():
        parser.add_argument(
            f'--{arg}',
            type=type(value),
            default=value,
            help=f'<{arg}>, default: {value}',
        )

    args = parser.parse_args()

    main(args)


