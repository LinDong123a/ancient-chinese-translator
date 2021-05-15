import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from data import AncientPairDataModule
from model import ModelInterface
from utils import get_args_by_parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--with_neptune", action="store_true")

    parser = pl.Trainer.add_argparse_args(parser)
    parser = AncientPairDataModule.add_data_args(parser)
    parser = ModelInterface.add_trainer_args(parser)

    parser = argparse.ArgumentParser(parents=[parser])
    args = parser.parse_args()

    if args.with_neptune:
        from pytorch_lightning.loggers.neptune import NeptuneLogger
        logger = NeptuneLogger(
            project_name="lds/ancient-chinese-translator",
        )
        logger.log_hyperparams(args)
    else:
        from pytorch_lightning.loggers.csv_logs import CSVLogger
        logger = CSVLogger("logs", name="ancient-chinese")

    # 加载训练数据
    data_module = AncientPairDataModule(
        **get_args_by_parser(args, AncientPairDataModule.parser),
    )

    # 提前加载数据，为了获取词表相关信息
    data_module.setup()

    model = ModelInterface(
        src_vocab=data_module.src_vocab,
        trg_vocab=data_module.trg_vocab,
        num_epoch=args.max_epochs,
        steps_per_epoch=len(data_module.train_dataloader()),
        model_config=get_args_by_parser(args, ModelInterface.model_cls.parser),
        **get_args_by_parser(args, ModelInterface.parser),
    )

    # 初始化Trainer的相关回调
    lr_monitor = LearningRateMonitor(logging_interval='step')
    model_checkpoint = ModelCheckpoint(
        monitor="valid/loss",
        dirpath=args.default_root_dir,
        mode="min",
    )

    trainer = pl.Trainer.from_argparse_args(
        args, logger=logger, callbacks=[lr_monitor, model_checkpoint],
    )
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)
