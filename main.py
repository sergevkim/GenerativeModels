from argparse import ArgumentParser
from pathlib import Path

from protostar.datamodules import ProtostarDataModule
from protostar.loggers import NeptuneLogger
from protostar.models import ProtostarModel
from protostar.trainer import Trainer

from config import Arguments


def main(args):
    model = ProtostarModel(
        learning_rate=args.learning_rate
        scheduler_gamma=args.scheduler_gamma,
        scheduler_step_size=args.scheduler_step_size,
        verbose=args.verbose,
    )
    datamodule = ProtostarDataModule(
        data_path=args.data_path,
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
    #parser = ArgumentParser()
    #args = parser.parse_args()
    args = Arguments()

    main(args)

