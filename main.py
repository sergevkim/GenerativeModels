from argparse import ArgumentParser
from pathlib import Path

import torch

from protostar.datamodules import ProtostarDataModule
from protostar.loggers import NeptuneLogger
from protostar.models import ProtostarModel
from protostar.trainer import Trainer


def main(args):
    model = ProtostarModel()
    datamodule = SpeechCommandsDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    datamodule.setup(val_ratio=args.val_ratio)

    logger = NeptuneLogger(
        api_key=None,
        project_name=None,
    )
    trainer = Trainer(
        logger=logger,
    )

    trainer.fit(
        model=model,
        datamodule=datamodule,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    args = parser.parse_args()

    main(args)

