import os
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as TF
import random


class SegmentationDataset(Dataset):
    """
    Dataset loader for 2D slices from ACDC.
    For each 3D volume, loads a set of central slices (default: 30).
    """

    def __init__(
        self,
        img_dir,
        lbl_dir,
        slice_axis=2,
        normalize=True,
        target_size=(256, 256),
        augment=False,
        num_slices_per_volume=30,
        use_cache=False
    ):
        self.slice_axis = slice_axis
        self.normalize = normalize
        self.target_size = target_size
        self.augment = augment
        self.num_slices_per_volume = num_slices_per_volume
        self.use_cache = use_cache

        self.img_paths = sorted(
            [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".nii.gz")]
        )
        self.lbl_paths = sorted(
            [os.path.join(lbl_dir, f) for f in os.listdir(lbl_dir) if f.endswith(".nii.gz")]
        )
        assert len(self.img_paths) == len(self.lbl_paths), "Number of images and labels must match!"

        # Optional cache for faster loading
        self.cache = {} if self.use_cache else None

        # Build (img_path, lbl_path, slice_index) list for central slices
        self.samples = []
        for img_path, lbl_path in zip(self.img_paths, self.lbl_paths):
            img_nii = nib.load(img_path)
            num_slices = img_nii.shape[self.slice_axis]

            if num_slices_per_volume is None or num_slices <= num_slices_per_volume:
                # Use all slices if None or fewer than limit
                slice_indices = list(range(num_slices))
            else:
                # Use only central subset
                center = num_slices // 2
                half = num_slices_per_volume // 2
                start = max(center - half, 0)
                end = min(center + half, num_slices)
                slice_indices = list(range(start, end))

            for s in slice_indices:
                self.samples.append((img_path, lbl_path, s))

    def __len__(self):
        return len(self.samples)

    def _load_volume(self, path):
        """Load from cache or disk."""
        if self.use_cache:
            if path not in self.cache:
                self.cache[path] = nib.load(path).get_fdata()
            return self.cache[path]
        else:
            return nib.load(path).get_fdata()

    def __getitem__(self, idx):
        img_path, lbl_path, s = self.samples[idx]

        # Load volume or from cache
        img = self._load_volume(img_path)
        lbl = self._load_volume(lbl_path)

        # Extract 2D slice
        img = np.take(img, s, axis=self.slice_axis)
        lbl = np.take(lbl, s, axis=self.slice_axis)

        # Normalize image
        if self.normalize:
            img = np.clip(img, np.percentile(img, 1), np.percentile(img, 99))
            img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)

        # To tensors
        img = torch.from_numpy(img).float().unsqueeze(0)  # [1,H,W]
        lbl = torch.from_numpy(lbl).long()                # [H,W]

        # Resize
        img = TF.resize(img, self.target_size, antialias=True)
        lbl = TF.resize(lbl.unsqueeze(0).float(), self.target_size,
                        interpolation=TF.InterpolationMode.NEAREST).squeeze(0).long()

        # ---------- Augmentations ----------
        if self.augment:
            # Random rotation (±15°)
            if random.random() < 0.5:
                angle = random.uniform(-15, 15)
                img = TF.rotate(img, angle)
                lbl = TF.rotate(lbl.unsqueeze(0).float(), angle,
                                interpolation=TF.InterpolationMode.NEAREST).squeeze(0).long()

            # # Random horizontal flip
            # if random.random() < 0.5:
            #     img = torch.flip(img, dims=[2])
            #     lbl = torch.flip(lbl, dims=[1])

            # # Random vertical flip
            # if random.random() < 0.5:
            #     img = torch.flip(img, dims=[1])
            #     lbl = torch.flip(lbl, dims=[0])

        return img, lbl


def get_train_val_loaders(
    img_dir,
    lbl_dir,
    batch_size=2,
    val_ratio=0.2,
    shuffle=True,
    seed=42,
    num_slices_per_volume=20
):
    """
    Returns train and validation DataLoaders.
    """
    dataset = SegmentationDataset(
        img_dir, lbl_dir,
        augment=True,
        num_slices_per_volume=num_slices_per_volume
    )


    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)

    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)
    val_ds.dataset.augment = False  # disable augmentation for val

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


if __name__ == "__main__":
    img_dir = "/media/ttoxopeus/datasets/nnUNet_raw/Dataset200_ACDC/imagesTr"
    lbl_dir = "/media/ttoxopeus/datasets/nnUNet_raw/Dataset200_ACDC/labelsTr"

    train_loader, val_loader = get_train_val_loaders(img_dir, lbl_dir)

    imgs, lbls = next(iter(train_loader))
    print("Image batch:", imgs.shape)
    print("Label batch:", lbls.shape)
    print("Unique labels:", torch.unique(lbls))
