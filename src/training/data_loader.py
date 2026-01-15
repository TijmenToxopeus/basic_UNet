import os
import torch
import numpy as np
import nibabel as nib
import torchio as tio
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

from src.utils.reproducibility import (
    seed_everything,
    seed_worker,
    make_generator,
)

# ============================================================
# --- TorchIO Augmentation Pipeline ---
# ============================================================
class SafeFlip(tio.Transform):
    """
    A TorchIO transform that replaces RandomFlip
    by using torch.flip (which never produces negative strides).
    """
    def __init__(self, axes=(1, 2), p=0.5):
        super().__init__(p=p)
        self.axes = axes

    def apply_transform(self, subject):
        for _, image in subject.get_images_dict().items():
            data = image.data
            torch_axes = tuple(a + 1 for a in self.axes)
            image.set_data(torch.flip(data, dims=torch_axes).clone())
        return subject


def get_torchio_augmentation_pipeline():
    return tio.Compose([
        tio.RandomAffine(scales=(0.9, 1.1), degrees=10, translation=5, p=0.5),
        tio.RandomElasticDeformation(num_control_points=7, max_displacement=4, p=0.3),
        tio.RandomGamma(log_gamma=(-0.5, 0.5), p=0.4),
        tio.RandomBiasField(coefficients=0.4, p=0.3),
        tio.RandomNoise(std=(0, 0.05), p=0.25),
        tio.RandomBlur(std=(0.1, 1.0), p=0.2),
    ])


def summarize_torchio_pipeline(pipeline):
    return [
        {"name": t.__class__.__name__, **{k: v for k, v in vars(t).items() if not k.startswith("_")}}
        for t in pipeline.transforms
    ]


# ============================================================
# --- Dataset (Patient/Volume-level split safe) ---
# ============================================================
class SegmentationDataset(Dataset):
    """
    Dataset loader for 2D slices from ACDC.

    Patient-level splitting is supported by passing a fixed list of
    (image_path, label_path) pairs into the dataset.
    """

    def __init__(
        self,
        img_lbl_pairs,
        slice_axis=2,
        normalize=True,
        target_size=(256, 256),
        augment=False,
        num_slices_per_volume=30,
        use_cache=False,
    ):
        self.img_lbl_pairs = list(img_lbl_pairs)
        self.slice_axis = slice_axis
        self.normalize = normalize
        self.target_size = target_size
        self.augment = augment
        self.num_slices_per_volume = num_slices_per_volume
        self.use_cache = use_cache

        self.cache = {} if self.use_cache else None

        # Build slice-level samples, but ONLY from the provided patient list
        self.samples = []
        for img_path, lbl_path in self.img_lbl_pairs:
            num_slices = nib.load(img_path).shape[self.slice_axis]

            if num_slices_per_volume is None or num_slices <= num_slices_per_volume:
                slice_indices = range(num_slices)
            else:
                center = num_slices // 2
                half = num_slices_per_volume // 2
                start = max(center - half, 0)
                end = min(center + half, num_slices)
                slice_indices = range(start, end)

            for s in slice_indices:
                self.samples.append((img_path, lbl_path, s))

        self.torchio_transform = get_torchio_augmentation_pipeline() if augment else None

    def __len__(self):
        return len(self.samples)

    def _load_volume(self, path):
        if self.use_cache:
            if path not in self.cache:
                self.cache[path] = nib.load(path).get_fdata()
            return self.cache[path]
        return nib.load(path).get_fdata()

    def __getitem__(self, idx):
        img_path, lbl_path, s = self.samples[idx]

        img = np.take(self._load_volume(img_path), s, axis=self.slice_axis)
        lbl = np.take(self._load_volume(lbl_path), s, axis=self.slice_axis)

        if self.normalize:
            img = np.clip(img, np.percentile(img, 1), np.percentile(img, 99))
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        img = torch.from_numpy(img).float().unsqueeze(0)  # [1, H, W]
        lbl = torch.from_numpy(lbl).long()                # [H, W]

        img = TF.resize(img, self.target_size, antialias=True)
        lbl = TF.resize(
            lbl.unsqueeze(0).float(),
            self.target_size,
            interpolation=TF.InterpolationMode.NEAREST,
        ).squeeze(0).long()

        if self.torchio_transform is not None:
            subject = tio.Subject(
                image=tio.ScalarImage(tensor=img.unsqueeze(0)),               # [1, 1, H, W]
                label=tio.LabelMap(tensor=lbl.unsqueeze(0).unsqueeze(0)),      # [1, 1, H, W]
            )
            subject = self.torchio_transform(subject)
            img = subject.image.data.squeeze(0)                               # [1, H, W]
            lbl = subject.label.data.squeeze().long()                         # [H, W]

        return img, lbl


# ============================================================
# --- Patient-level split helper ---
# ============================================================
def _collect_img_lbl_pairs(img_dir, lbl_dir):
    img_paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".nii.gz")])
    lbl_paths = sorted([os.path.join(lbl_dir, f) for f in os.listdir(lbl_dir) if f.endswith(".nii.gz")])
    assert len(img_paths) == len(lbl_paths), "Number of images and labels must match!"
    return list(zip(img_paths, lbl_paths))


def _patient_level_split(pairs, val_ratio, seed):
    """
    Split at the patient/volume level.
    """
    n_total = len(pairs)
    n_val = int(round(n_total * val_ratio))
    n_train = n_total - n_val

    g = make_generator(seed)
    perm = torch.randperm(n_total, generator=g).tolist()

    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    train_pairs = [pairs[i] for i in train_idx]
    val_pairs = [pairs[i] for i in val_idx]
    return train_pairs, val_pairs


# ============================================================
# --- DataLoader Builder (Patient-level split) ---
# ============================================================
def get_train_val_loaders(
    img_dir,
    lbl_dir,
    batch_size=2,
    val_ratio=0.2,
    shuffle=True,
    seed=42,
    num_slices_per_volume=20,
    num_workers=0,
    deterministic=False,
):
    """
    Returns train and validation DataLoaders with PATIENT-LEVEL splitting.
    """

    seed_everything(seed, deterministic=deterministic)

    pairs = _collect_img_lbl_pairs(img_dir, lbl_dir)
    train_pairs, val_pairs = _patient_level_split(pairs, val_ratio=val_ratio, seed=seed)

    train_dataset = SegmentationDataset(
        train_pairs,
        augment=True,
        num_slices_per_volume=num_slices_per_volume,
    )
    val_dataset = SegmentationDataset(
        val_pairs,
        augment=False,
        num_slices_per_volume=num_slices_per_volume,
    )

    loader_gen = make_generator(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=loader_gen,
        persistent_workers=num_workers > 0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=loader_gen,
        persistent_workers=num_workers > 0,
        pin_memory=True,
    )

    augmentation_summary = (
        summarize_torchio_pipeline(train_dataset.torchio_transform)
        if train_dataset.torchio_transform is not None
        else None
    )

    return train_loader, val_loader, augmentation_summary


# ============================================================
# --- Debug / Quick Test ---
# ============================================================
if __name__ == "__main__":
    img_dir = "/media/ttoxopeus/datasets/nnUNet_raw/Dataset200_ACDC/imagesTr"
    lbl_dir = "/media/ttoxopeus/datasets/nnUNet_raw/Dataset200_ACDC/labelsTr"

    train_loader, val_loader, aug = get_train_val_loaders(
        img_dir,
        lbl_dir,
        seed=42,
        num_workers=0,
        deterministic=True,
    )

    imgs, lbls = next(iter(train_loader))
    print("Image batch:", imgs.shape)
    print("Label batch:", lbls.shape)
    print("Unique labels:", torch.unique(lbls))
    print("Augmentations:", aug)
