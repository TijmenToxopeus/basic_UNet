import os
import torch
import numpy as np
import nibabel as nib
import torchio as tio
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as TF
import random


# ============================================================
# --- TorchIO Augmentation Pipeline ---
# ============================================================
def get_torchio_augmentation_pipeline():
    """Return a TorchIO augmentation pipeline for MRI-like images."""
    return tio.Compose([
        # # Resize to larger resolution first
        # tio.Resize((1, 320, 320)),

        # # Central crop to focus on heart
        # tio.CropOrPad((1, 256, 256)),
        tio.RandomElasticDeformation(num_control_points=7, max_displacement=3, p=0.3),  # elastic deformation
        tio.RandomAffine(scales=(0.9, 1.1), degrees=10,              # rotation/scaling
                         translation=5, p=0.5),
        tio.RandomNoise(mean=0, std=(0, 0.05), p=0.25),              # Gaussian noise
        tio.RandomBiasField(coefficients=0.3, p=0.3),                 # MRI intensity bias
        tio.RandomGamma(log_gamma=(-0.3, 0.3), p=0.3),                # gamma correction
    ])



def summarize_torchio_pipeline(pipeline):
    summary = []
    for transform in pipeline.transforms:
        params = {k: v for k, v in vars(transform).items() if not k.startswith('_')}
        summary.append({"name": transform.__class__.__name__, **params})
    return summary


# ============================================================
# --- Dataset Class ---
# ============================================================
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

        # Preload slice indices
        self.samples = []
        for img_path, lbl_path in zip(self.img_paths, self.lbl_paths):
            img_nii = nib.load(img_path)
            num_slices = img_nii.shape[self.slice_axis]

            if num_slices_per_volume is None or num_slices <= num_slices_per_volume:
                slice_indices = list(range(num_slices))
            else:
                center = num_slices // 2
                half = num_slices_per_volume // 2
                start = max(center - half, 0)
                end = min(center + half, num_slices)
                slice_indices = list(range(start, end))

            for s in slice_indices:
                self.samples.append((img_path, lbl_path, s))

        # Initialize TorchIO augmentation pipeline if requested
        self.torchio_transform = get_torchio_augmentation_pipeline() if augment else None

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

        # Convert to torch tensors
        img = torch.from_numpy(img).float().unsqueeze(0)  # [1, H, W]
        lbl = torch.from_numpy(lbl).long()                # [H, W]

        # Resize both
        img = TF.resize(img, self.target_size, antialias=True)
        lbl = TF.resize(
            lbl.unsqueeze(0).float(),
            self.target_size,
            interpolation=TF.InterpolationMode.NEAREST
        ).squeeze(0).long()

        # ---------- TorchIO Augmentations ----------
        # if self.torchio_transform is not None:
        #     subject = tio.Subject(
        #         image=tio.ScalarImage(tensor=img.unsqueeze(0)),  # add batch dim
        #         label=tio.LabelMap(tensor=lbl.unsqueeze(0))
        #     )
        #     transformed = self.torchio_transform(subject)
        #     img = transformed.image.data.squeeze(0)
        #     lbl = transformed.label.data.squeeze(0).long()
        if self.torchio_transform is not None:
            subject = tio.Subject(
                image=tio.ScalarImage(tensor=img.unsqueeze(0)),               # [1, 1, H, W]
                label=tio.LabelMap(tensor=lbl.unsqueeze(0).unsqueeze(0))      # [1, 1, H, W]
            )
            transformed = self.torchio_transform(subject)
            img = transformed.image.data.squeeze(0)                           # → [1, H, W]
            lbl = transformed.label.data.squeeze().long()                     # → [H, W]


        return img, lbl


# ============================================================
# --- DataLoader Builder ---
# ============================================================
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
    Uses two separate dataset instances so that:
      - training data has augmentations,
      - validation data does not.
    Also returns an augmentation summary for logging.
    """

    # --- Create two *independent* datasets ---
    train_dataset = SegmentationDataset(
        img_dir,
        lbl_dir,
        augment=True,
        num_slices_per_volume=num_slices_per_volume
    )
    val_dataset = SegmentationDataset(
        img_dir,
        lbl_dir,
        augment=False,  # explicitly disable augmentations
        num_slices_per_volume=num_slices_per_volume
    )

    # --- Split indices reproducibly ---
    val_size = int(len(train_dataset) * val_ratio)
    train_size = len(train_dataset) - val_size
    generator = torch.Generator().manual_seed(seed)

    train_subset, _ = random_split(train_dataset, [train_size, val_size], generator=generator)
    _, val_subset = random_split(val_dataset, [train_size, val_size], generator=generator)

    # --- Build DataLoaders ---
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    # --- Summarize augmentation pipeline (for logging) ---
    augmentation_summary = None
    if hasattr(train_dataset, "torchio_transform") and train_dataset.torchio_transform is not None:
        augmentation_summary = summarize_torchio_pipeline(train_dataset.torchio_transform)

    return train_loader, val_loader, augmentation_summary



# def get_train_val_loaders(
#     img_dir,
#     lbl_dir,
#     batch_size=2,
#     val_ratio=0.2,
#     shuffle=True,
#     seed=42,
#     num_slices_per_volume=20
# ):
#     """
#     Returns train and validation DataLoaders.
#     """
#     dataset = SegmentationDataset(
#         img_dir,
#         lbl_dir,
#         augment=True,
#         num_slices_per_volume=num_slices_per_volume
#     )

#     val_size = int(len(dataset) * val_ratio)
#     train_size = len(dataset) - val_size
#     generator = torch.Generator().manual_seed(seed)

#     train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)
#     val_ds.dataset.augment = False
#     val_ds.dataset.torchio_transform = None  # disable augmentations for val

#     train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
#     val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

#         # --- Summarize augmentation pipeline (for logging) ---
#     augmentation_summary = None
#     if hasattr(dataset, "torchio_transform") and dataset.torchio_transform is not None:
#         augmentation_summary = summarize_torchio_pipeline(dataset.torchio_transform)

#     return train_loader, val_loader, augmentation_summary


# ============================================================
# --- Debug / Quick Test ---
# ============================================================
if __name__ == "__main__":
    img_dir = "/media/ttoxopeus/datasets/nnUNet_raw/Dataset200_ACDC/imagesTr"
    lbl_dir = "/media/ttoxopeus/datasets/nnUNet_raw/Dataset200_ACDC/labelsTr"

    train_loader, val_loader = get_train_val_loaders(img_dir, lbl_dir)

    imgs, lbls = next(iter(train_loader))
    print("Image batch:", imgs.shape)
    print("Label batch:", lbls.shape)
    print("Unique labels:", torch.unique(lbls))
