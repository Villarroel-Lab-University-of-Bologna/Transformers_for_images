from torch.utils.data import Dataset, DataLoader
import torch
from . import knutills as kutil

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TrainingImageDataset(Dataset):
    def __init__(self, image_data, labels, processor):
        self.images = image_data
        self.labels = labels
        self.processor = processor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].convert("RGB")
        processed_image = self.processor(
            images=image, return_tensors="pt"
        ).pixel_values.squeeze(0)
        label = self.labels[idx]
        return processed_image, label


class PredictionImageDataset(Dataset):
    def __init__(self, image_data, processor):
        self.images = image_data
        self.processor = processor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images.iloc[idx].convert("RGB")
        processed_image = self.processor(
            images=image, return_tensors="pt"
        ).pixel_values.squeeze(0)
        return processed_image


class ViTModelSelection(kutil.knext.EnumParameterOptions):
    ViT = (
        "ViT",
        "google/vit-base-patch16-224",
    )
    SWIN = (
        "Swin",
        "microsoft/swin-base-patch4-window7-224",
    )
    PYRAMID = (
        "Pyramid Transformer",
        "Zetatech/pvt-medium-224",
    )

    @classmethod
    def get_default(cls):
        return cls.ViT
