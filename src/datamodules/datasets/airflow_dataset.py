import torch
import os
import torch.utils.data as data
import numpy as np


class AirflowDataset(data.Dataset):
    def __init__(
        self, data_list, data_dir, mean_std, stage, sample_size=32000
    ) -> None:
        self.data_list = data_list
        self.data_dir = data_dir
        self.mean_std = mean_std
        self.sample_size = sample_size
        self.stage = stage
        self.data_dir = self.data_dir
        super().__init__()

    def __getitem__(self, index):
        data_sample = torch.load(
            os.path.join(self.data_dir, f"{self.data_list[index]}.pth"),
            weights_only=False,
        )
        data_sample.x = (data_sample.x - self.mean_std[0]) / (
            self.mean_std[1] + 1e-9
        )
        data_sample.y = (data_sample.y - self.mean_std[2]) / (
            self.mean_std[3] + 1e-9
        )
        idx = np.random.randint(data_sample.x.size(0), size=self.sample_size)
        x = (
            data_sample.x
            if self.stage in ["test", "predict"]
            else data_sample.x[idx]
        )
        y = (
            data_sample.y
            if self.stage in ["test", "predict"]
            else data_sample.y[idx]
        )
        surf = (
            data_sample.surf
            if self.stage in ["test", "predict"]
            else data_sample.surf[idx]
        )
        pos = x[:, :2]
        x = x[:, 2:]
        return pos, x, y, surf

    def __len__(self):
        return len(self.data_list)
