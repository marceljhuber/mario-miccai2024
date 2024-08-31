import numpy as np
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
import random

from torchvision.transforms import InterpolationMode
from PIL import Image

###################################################################################
transformV1 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
###################################################################################
transformV2 = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
###################################################################################
class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean


transformV3 = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    AddGaussianNoise(mean=0.0, std=0.1)
])
###################################################################################
transformV4 = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
###################################################################################
transformV5 = transforms.Compose([
    transforms.RandomCrop(224),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
###################################################################################
transformV5 = transforms.Compose([
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
###################################################################################
class ConditionalTransform:
    def __init__(self, base_transform, special_transform, probability=0.1):
        self.base_transform = base_transform
        self.special_transform = special_transform
        self.probability = probability

    def __call__(self, img):
        if random.random() < self.probability:
            return self.special_transform(img)
        else:
            return self.base_transform(img)


transformV6 = ConditionalTransform(transformV1, transformV5, probability=0.15)
###################################################################################
transformV7 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    AddGaussianNoise(mean=0.0, std=0.1),
])
###################################################################################
class HighPassFilterV8:
    def __init__(self, device, kernel_size=3):
        self.kernel_size = kernel_size
        self.device = device
        # Define the high-pass filter kernel
        self.kernel = torch.tensor([[-1, -1, -1],
                                    [-1, 1, -1],
                                    [-1, -1, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    def __call__(self, img):
        if isinstance(img, Image.Image):
            img = transforms.ToTensor()(img).unsqueeze(0).to(self.device)

        # Make it grayscale
        img = torch.mean(img, 0).unsqueeze(0)

        # Apply the high-pass filter
        img = F.conv2d(img, self.kernel, padding=1)

        # Make it RGB again
        img = img.repeat(3, 1, 1)

        # Return the tensor
        return img


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transformV8 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    HighPassFilterV8(device=device)
])
###################################################################################
class HighPassFilterV9:
    def __init__(self, device, kernel_size=3):
        self.kernel_size = kernel_size
        self.device = device
        # Define the high-pass filter kernel
        self.kernel = torch.tensor([[-0.5, -0.5, -0.5],
                                    [-0.5, 0.5, -0.5],
                                    [-0.5, -0.5, -0.5]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

    def __call__(self, img):
        if isinstance(img, Image.Image):
            img = transforms.ToTensor()(img).unsqueeze(0).to(self.device)

        # Make it grayscale
        img = torch.mean(img, 0).unsqueeze(0)

        # Apply the high-pass filter
        img = img.to(device)
        img = F.conv2d(img, self.kernel, padding=1)

        # Make it RGB again
        img = img.repeat(3, 1, 1)

        # Return the tensor
        return img


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transformV9 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    HighPassFilterV9(device=device)
])
###################################################################################
transformV10 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


###################################################################################
class ConditionalTransformV11:
    def __init__(self, base_transform, special_transform, probability=0.1):
        self.base_transform = base_transform
        self.special_transform = special_transform
        self.probability = probability

    def __call__(self, img):
        if random.random() < self.probability:
            return self.special_transform(img)
        else:
            return self.base_transform(img)


transformV11 = ConditionalTransformV11(transformV1, transformV10, probability=0.15)
###################################################################################
transformV12 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda image: image.convert('RGB')),
    transforms.AugMix(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])
###################################################################################
transformV13 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda image: image.convert('RGB')),
    transforms.AugMix(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
###################################################################################
transformV14 = transforms.Compose([
    transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(240),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
###################################################################################
transformV15 = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.333))
])
###################################################################################
transformV16 = transforms.Compose([
    transforms.RandomRotation(degrees=10),  # Random rotation between -30 to +30 degrees
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),  # Random translation, scaling, and shearing
    transforms.RandomCrop(size=(240, 240)),  # Random crop to size 240x240
    transforms.ColorJitter(brightness=0.2),  # Add brightness noise with a factor of 0.2
    transforms.ToTensor()  # Convert PIL Image to tensor
])
###################################################################################
class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0, p=0.5):
        """
        Args:
            mean (float): Mean of the Gaussian noise.
            std (float): Standard deviation of the Gaussian noise.
            p (float): Probability of applying the transform.
        """
        self.mean = mean
        self.std = std
        self.p = p
    
    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Input tensor to which noise will be added.
            
        Returns:
            Tensor: Tensor with Gaussian noise added.
        """
        if torch.rand(1).item() < self.p:
            noise = torch.normal(mean=self.mean, std=self.std, size=tensor.size())
            tensor = tensor + noise
        return tensor

transformV17 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    AddGaussianNoise(mean=0.0, std=0.1, p=0.5),
])
###################################################################################
class AddSpeckleNoise:
    def __init__(self, prob=0.5, intensity=0.1):
        self.prob = prob
        self.intensity = intensity

    def __call__(self, img):
        if random.random() < self.prob:
            noise = torch.randn_like(img) * (self.intensity * random.random())
            return img * (1 + noise)
        return img


transformV18 = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.333)),
        transforms.ToTensor(),
        AddSpeckleNoise(prob=0.5, intensity=0.1),  # Add speckle noise
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Grayscale(num_output_channels=3),
    ])
###################################################################################
# transformV1 with Grayscale
transformV19 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.Grayscale(num_output_channels=3),
])
###################################################################################
class AddSpeckleNoise(object):
    def __init__(self, mean=0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            img_np = img.numpy()
        else:
            img_np = np.array(img)

        noise = np.random.normal(self.mean, self.std, img_np.shape)
        noisy_img = img_np + img_np * noise

        return torch.tensor(noisy_img).clamp(0, 1) if isinstance(img, torch.Tensor) else transforms.ToPILImage()(torch.tensor(noisy_img).clamp(0, 1))

# transformV1 with Speckle Noise
transformV20 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    AddSpeckleNoise(mean=0, std=0.1),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
###################################################################################
# transformV1 with new mean, std
transformV21 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.35], std=[0.5]),
])
###################################################################################
class AddSpeckleNoise:
    def __init__(self, prob=0.5, intensity=0.1):
        self.prob = prob
        self.intensity = intensity

    def __call__(self, img):
        if random.random() < self.prob:
            noise = torch.randn_like(img) * (self.intensity * random.random())
            return img * (1 + noise)
        return img

# transformV1 with Grayscale
transformV22 = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.333)),
        transforms.ToTensor(),
        AddSpeckleNoise(prob=0.5, intensity=0.15),  # Add speckle noise
        transforms.Normalize(mean=[0.35], std=[0.5]),
    ])
###################################################################################
transformV23 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.35], std=[0.75]),
])
###################################################################################
transformV24 = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.35], std=[0.75]),
])