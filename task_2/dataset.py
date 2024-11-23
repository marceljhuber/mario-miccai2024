# encoding: utf-8

from torch.utils.data import Dataset
import torch

from torch_device import get_device
from typing import Optional
import pandas as pd
import os


class NinetyDaysLatentMatching(Dataset):
    """
    PyTorch dataset loading two consecutive OCT volumes with their
    fundus images.
    """
    def __init__(
            self,
            data: pd.DataFrame,
            device: torch.device,
            tensor_path: Optional[str],
    ) -> None:
        """
        Pass either a path to img_path or tensor_path to switch between loading
        images and tensors. Keep the other one None.

        :param data:
        :param tensor_path: str path to tensor dataset
        :param device: torch device to load the stored tensors from
        """

        self.data: pd.DataFrame = data
        self.tensor_path: str = tensor_path
        self.device: torch.device = device

    def __getitem__(self, index: int):
        sample = self.data.iloc[index]

        oct_t0 = torch.load(
            os.path.join(self.tensor_path, sample['image_at_ti'][:-4] + '.pt'),
            map_location=self.device
        )
        oct_t90 = torch.load(
            os.path.join(self.tensor_path, sample['image_at_ti+90'][:-4] + '.pt'),
            map_location=self.device
        )

        return oct_t0, oct_t90, torch.zeros(0,), torch.zeros(0,), {}

    def __len__(self) -> int:
        return len(self.data)


if __name__ == '__main__':
    # Example usage:
    device = get_device()

    ds = NinetyDaysLatentMatching(
        data=pd.read_csv(os.path.join('..', 'data_1', '90_days_data.csv')),
        tensor_path=os.path.join('..', 'data_1', 'data_task1_fe', 'train'),
        device=device if device == torch.device('cuda') else torch.device('cpu'),
    )
    print(ds[0][0].shape)
