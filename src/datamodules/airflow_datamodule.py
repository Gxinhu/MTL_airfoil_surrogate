import torch
from lightning import LightningDataModule
import numpy as np

from torch.utils.data import DataLoader
from .datasets.airflow import AirflowDatasetBuilder
import json
import os


class AirflowDataModule(LightningDataModule):
    """
    Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset = None,
        data_dir: str = "data/",
        task: str = "full",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        regenerate: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)
        self.batch_size = batch_size
        self.task = task
        self.data_dir = data_dir

        with open(f"{data_dir}/manifest.json", encoding="UTF-8") as fp:
            manifest = json.load(fp)
        manifest_train = manifest[f"{task}_train"]
        self.test_list = (
            manifest[f"{task}_test"]
            if task != "scarce"
            else manifest["full_test"]
        )
        self.n = 0.1
        self.train_list = manifest_train
        self.processed_path = os.path.join(data_dir, "torch_data")
        self.corf_norm = None

        # data transformations
        self.data_train = None
        self.data_val = None
        self.data_test = None
        self.val_list = None
        self.dataset = dataset

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""

    def setup(self, stage: str):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning twice for `trainer.fit()` and `trainer.test()`, so be careful if you do a random split!
        The `stage` can be used to differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """

        # load datasets only if they're not loaded already
        if self.data_train or self.data_val or self.data_test:
            return

        if (
            not os.path.exists(self.processed_path)
            and not self.hparams.regenerate
        ):
            if not os.path.exists(self.processed_path):
                os.mkdir(self.processed_path)
            AirflowDatasetBuilder(
                self.train_list,
                sample=None,
                base_folder=self.data_dir,
                save_folder=self.processed_path,
            )
            AirflowDatasetBuilder(
                self.test_list,
                sample=None,
                base_folder=self.data_dir,
                save_folder=self.processed_path,
            )

        val_idx = np.zeros(len(self.train_list), dtype=bool)
        val_idx[: int(self.n * len(self.train_list))] = True
        np.random.shuffle(val_idx)
        n = int(self.n * len(self.train_list))
        self.val_list = self.train_list[-n:]
        self.train_list = self.train_list[:-n]

        n = 0
        mean_input = 0.0
        M2_input = 0.0
        mean_output = 0.0
        M2_output = 0.0
        for filename in self.train_list:
            data = torch.load(
                os.path.join(self.processed_path, f"{filename}.pth"),
                weights_only=False,
            )
            n_new = len(data.x)
            n += n_new

            new_mean_input = torch.mean(data.x, dim=0)
            mean_input = mean_input + (new_mean_input - mean_input) * (
                n_new / n
            )
            M2_input += (
                torch.sum((data.x - new_mean_input) ** 2, dim=0)
                + n_new * (mean_input - new_mean_input) ** 2
            )
            new_mean_output = torch.mean(data.y, dim=0)
            mean_output = mean_output + (new_mean_output - mean_output) * (
                n_new / n
            )
            M2_output += (
                torch.sum((data.y - new_mean_output) ** 2, dim=0)
                + n_new * (mean_output - new_mean_output) ** 2
            )

        std_input = np.sqrt(M2_input / n)
        std_output = np.sqrt(M2_output / n)
        self.mean_std = (mean_input, std_input, mean_output, std_output)
        self.data_train = self.dataset(
            data_list=self.train_list,
            mean_std=self.mean_std,
            data_dir=self.processed_path,
            stage="train",
        )
        self.data_val = self.dataset(
            data_list=self.val_list,
            mean_std=self.mean_std,
            data_dir=self.processed_path,
            stage="val",
        )
        self.data_test = self.dataset(
            data_list=self.test_list,
            mean_std=self.mean_std,
            data_dir=self.processed_path,
            stage="test",
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
