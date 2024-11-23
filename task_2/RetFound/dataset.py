# encoding: utf-8

"""
Dataset class returning two consecutive OCT volumes with their
corresponding fundus localizer.

Example usage:
transform = v2.Compose([
    v2.ToImage(),
    v2.Grayscale(num_output_channels=1),
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize((224, 224)),
])

ds = TwoVisitOCTDataset(
    t_transform=transform,
    data_path='../data_1/train_splits',
    img_path='../data_1/data_task1/train',
    return_csv=False,
)
x = ds[1]
"""

from torchvision.transforms import v2
from torchvision.io import read_image
from torch.utils.data import Dataset
import torch

from utils.torch_device import get_device
from typing import List, Optional
from glob import glob
import pandas as pd
import os


class TwoVisitOCTDataset(Dataset):
    """
    PyTorch dataset loading two consecutive OCT volumes with their
    fundus images.
    """
    def __init__(self, t_transform: v2.Compose, data_path: str, device: torch.device,
                 img_path: Optional[str], tensor_path: Optional[str], return_csv: bool = False) -> None:
        """
        Pass either a path to img_path or tensor_path to switch between loading
        images and tensors. Keep the other one None.

        :param t_transform: v2.Compose preprocessing applied to loaded images
        :param data_path: str path to the folder with CSV files from 'split_csv.py'
        :param img_path: str path to image dataset
        :param tensor_path: str path to tensor dataset
        :param device: torch device to load the stored tensors from
        :param return_csv: bool whether to return the information from the CSV file
        """

        self.data_path: str = data_path
        self.img_path: str = img_path
        self.tensor_path: str = tensor_path
        self.return_csv: bool = return_csv
        self.transforms: v2.Compose = t_transform
        self.files: List[str] = glob(os.path.join(data_path, '**', '*.csv'), recursive=True)
        self.device: torch.device = device

    def __getitem__(self, index: int):
        df: pd.DataFrame = pd.read_csv(self.files[index])

        # Load images in case an image path is give, otherwise the stored tensors.
        if self.img_path:
            oct_t = self._make_img_volume(df.loc[:, 'image_at_ti'].to_list())
            oct_t1 = self._make_img_volume(df.loc[:, 'image_at_ti+1'].to_list())
            fundus = self._make_img_volume([df.loc[0, 'LOCALIZER_at_ti'], df.loc[0, 'LOCALIZER_at_ti+1']])
        else:
            oct_t = self._make_tensor_volume(df.loc[:, 'image_at_ti'].to_list())
            oct_t1 = self._make_tensor_volume(df.loc[:, 'image_at_ti+1'].to_list())
            fundus = self._make_tensor_volume([df.loc[0, 'LOCALIZER_at_ti'], df.loc[0, 'LOCALIZER_at_ti+1']])

        # In case the df has no label information (val) make a dummy variable.
        if 'label' in df:
            y = torch.tensor(df.loc[:, 'label'])
        else:
            y = torch.zeros(size=(oct_t.shape[0], )).long()

        return oct_t, oct_t1, fundus, y, df.to_dict() if self.return_csv else {}

    def _make_img_volume(self, img_paths: List[str]) -> torch.Tensor:
        """
        Loads b_scans/fundus images, processes them and concatenates
        them into a 3D tensor (representing the oct volume for b_scans).

        :param img_paths: List[str] list containing image names
        :return: torch.Tensor 3D tensor of the oct volume
        """

        scan_tensor = []
        for img_path in img_paths:
            img = read_image(os.path.join(self.img_path, img_path))
            img = self.transforms(img)
            scan_tensor.append(img)

        return torch.cat(scan_tensor, dim=-2)

    def _make_tensor_volume(self, tensor_paths: List[str]) -> torch.Tensor:
        """
        Loads feature extractions from images and concatenates
        them into a 3D tensor (representing the oct volume for b_scans).

        :param tensor_paths: List[str] list containing image names
        :return: torch.Tensor 3D tensor of the oct volume
        """

        scan_tensor = []
        for tensor_path in tensor_paths:
            tensor = torch.load(os.path.join(self.tensor_path, tensor_path[:-4] + '.pt'), map_location=self.device)
            scan_tensor.append(tensor)

        return torch.stack(scan_tensor, dim=0)

    def __len__(self) -> int:
        return len(self.files)


if __name__ == '__main__':
    # Example usage:
    transform = v2.Compose([
        v2.ToImage(),
        v2.Grayscale(num_output_channels=1),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((224, 224)),
    ])

    device = get_device()

    ds = TwoVisitOCTDataset(
        t_transform=transform,
        data_path=os.path.join('..', 'data_1', 'train_splits'),
        img_path=None,  # os.path.join('..', 'data_1', 'data_task1', 'train'),
        tensor_path=os.path.join('..', 'data_1', 'data_task1_fe', 'train'),
        return_csv=False,
        device=device if device == torch.device('cuda') else torch.device('cpu'),
    )
    x = ds[1]
    print(x[0].shape, x[1].shape, x[2].shape, x[3].shape)
