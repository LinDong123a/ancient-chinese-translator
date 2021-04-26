import pytorch_lightning as pl
from pytorch_lightning.loggers.neptune import NeptuneLogger

from data import AncientPairDataModule
from model import ModelInterface

neptune_logger = NeptuneLogger(
    project_name="lds/ancient-chinese-translator",
)


if __name__ == "__main__":
    data_module = AncientPairDataModule(batch_size=4, data_dir="./dataset")

    data_module.setup()

    model = ModelInterface("GRU", data_module.src_vocab, data_module.trg_vocab)

    print(len(data_module.train_dataset))
    print(len(data_module.test_dataset))
    print(len(data_module.valid_dataset))

    trainer = pl.Trainer(logger=neptune_logger, gpus=1, precision=16, log_every_n_steps=1)
    trainer.fit(model, datamodule=data_module)

    trainer.test(model, datamodule=data_module)
