# encoding: utf-8

"""
Feature extractor for batches b-scans or an OCT volume.
Can also be applied to fundus images.

# Example usage:
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

x = ds[1]  # first entry in the dataset
x = x[0]   # first OCT volume
x = x[:2]  # first two b-scans

fe = RetFoundFeatureExtractor(
    ckpt_path='../ckpts/RetFoundFeatureExtractor/RetFound_oct_weights.pth',
    device=torch.device('cpu'),
)

feats = fe.extract_features(x)
print(feats.shape)
"""

from dataset import TwoVisitOCTDataset
import models_vit

from torchvision.transforms import v2
from typing import List, Optional
import torch
import os


class RetFoundFeatureExtractor:
    """
    Feature extractor for batches b-scans or an OCT volume.
    Can also be applied to fundus images.
    """
    def __init__(
            self,
            ckpt_path: str,
            device: torch.device,
            mean: Optional[List[float]] = None,
            std: Optional[List[float]] = None,
    ) -> None:
        """
        :param ckpt_path: str path to the pretrained RetFound model
            -> download https://drive.google.com/file/d/1m6s7QYkjyjJDlpEuXm7Xp3PmjN-elfW2/view?usp=sharing
            for OCT weights
            -> download https://drive.google.com/file/d/1l62zbWUFTlp214SvK6eMwPQZAzcwoeBE/view?usp=sharing
            for fundus weights
        :param device: torch device to run inference on
        :param mean: List[float] used for normalization
        :param std: List[float] used for normalization
        """

        self.ckpt_path: str = ckpt_path
        self.device: torch.device = device
        self.model: torch.nn.Module = self._prepare_model()

        self.transform: v2.Compose = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((224, 224)),
            v2.Normalize(
                mean=mean if mean else [0.485, 0.456, 0.406],
                std=std if std else [0.229, 0.224, 0.225],
            ),
        ])

    def _prepare_model(self) -> torch.nn.Module:

        model = models_vit.__dict__['vit_large_patch16'](
            img_size=224,
            num_classes=5,
            drop_path_rate=0,
            global_pool=True,
        ).to(self.device)

        checkpoint = torch.load(self.ckpt_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)

        return model

    def __call__(self, x: torch.Tensor) -> torch.Tensor:

        # Add a channel dim and repeat the image 3 times along this dim
        x = x[:, None, :, :].repeat(1, 3, 1, 1)
        x = self.transform(x).to(self.device, non_blocking=True)

        return self.model.forward_features(x.float())


if __name__ == '__main__':

    # Example usage:
    transform = v2.Compose([
        v2.ToImage(),
        v2.Grayscale(num_output_channels=1),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((224, 224)),
    ])

    ds = TwoVisitOCTDataset(
        t_transform=transform,
        data_path=os.path.join('..', 'data_1', 'train_splits'),
        img_path=os.path.join('..', 'data_1', 'data_task1', 'train'),
        return_csv=False,
        device=torch.device('cpu'),
        tensor_path=os.path.join('..', 'data_1', 'data_task1_fe', 'train'),
    )

    x = ds[1]  # first entry in the dataset
    x = x[0]   # first OCT volume
    x = x[:2]  # first two b-scans

    fe = RetFoundFeatureExtractor(
        ckpt_path=os.path.join('..', 'ckpts', 'RetFoundFeatureExtractor', 'RetFound_oct_weights.pth'),
        device=torch.device('cpu'),
    )

    feats = fe(x)
    print(feats.shape)
