"""Data: annotation files, species names, stratified split, transforms and DataLoaders.

Pipeline: train.txt / test.txt (file name + label) -> DataFrames with the full image path -> a stratified
80/20 train / validation split -> PyTorch DataLoaders that read, augment and normalise the images.
"""
import os
import re
from dataclasses import dataclass

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from . import config as C


# ---------------------------------------------------------------------------- annotations
def read_annotations(path):
    """Each line is "<image name> <class label>". Returns a DataFrame with columns file, label."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                name, label = line.rsplit(" ", 1)      # split on the LAST space only
                rows.append((name, int(label)))
    return pd.DataFrame(rows, columns=["file", "label"])


def index_images(root):
    """Map file name -> full path, whatever folder layout the zip was extracted with."""
    idx = {}
    for d, _, files in os.walk(root):
        if "__MACOSX" in d:                             # skip macOS zip metadata
            continue
        for fn in files:
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                idx[fn] = os.path.join(d, fn)
    return idx


def species_name(fname):
    """Black_footed_Albatross_0019_416160254.jpg -> "Black footed Albatross".

    91 training files end in "_xxx.jpg" instead of a number, so the last token is matched as [^_]+.
    """
    return re.sub(r"_\d+_[^_]+\.jpg$", "", fname).replace("_", " ")


def class_names_from(df, num_classes):
    """One species name per label; fails loudly if a label maps to more than one name."""
    names = df.groupby("label").file.apply(lambda s: sorted(set(map(species_name, s))))
    bad = names[names.str.len() != 1]
    assert bad.empty, f"Labels with inconsistent species names:\n{bad}"
    return [names[i][0] for i in range(num_classes)]


@dataclass
class BirdData:
    train: pd.DataFrame        # 80% of train.txt, used for gradient updates
    val: pd.DataFrame          # 20% of train.txt, used for every decision (early stopping, selection)
    test: pd.DataFrame         # test.txt, used only for the final reported numbers
    full_train: pd.DataFrame   # all of train.txt (train + val)
    class_names: list

    @property
    def num_classes(self):
        return len(self.class_names)


def load_data(seed=C.SEED):
    """Read both annotation files, find every image on disk and split train.txt into train / validation.

    The test set is kept completely separate: it is only used by evaluate.py for the final numbers.
    The same seed always gives the same split, so every experiment uses exactly the same images.
    """
    full = read_annotations(C.TRAIN_TXT)
    test = read_annotations(C.TEST_TXT)
    # attach the full path of every image (NaN if a file listed in the .txt is missing on disk)
    full["path"] = full.file.map(index_images(C.TRAIN_IMG_DIR))
    test["path"] = test.file.map(index_images(C.TEST_IMG_DIR))
    assert full.path.notna().all(), f"Some training images are missing from {C.TRAIN_IMG_DIR}"
    assert test.path.notna().all(), f"Some test images are missing from {C.TEST_IMG_DIR}"

    num_classes = int(full.label.max()) + 1          # labels are 0 .. 199
    class_names = class_names_from(full, num_classes)  # species names, read from the file names

    # Stratified split so every class appears in both train and validation
    train, val = train_test_split(full, test_size=C.VAL_FRACTION, stratify=full.label,
                                  random_state=seed)
    return BirdData(train, val, test, full, class_names)


# ---------------------------------------------------------------------------- transforms
def make_transforms(img_size, mean=C.MEAN, std=C.STD, augment="basic"):
    """(train transform, eval transform) for a square input of img_size pixels.

    augment="none"   : training images get the eval preprocessing (no randomness), used for baselines
    augment="basic"  : random resized crop (60-100% of the image), flip, rotation, colour jitter, random erasing
    augment="strong" : wider random resized crop (25-100%), flip, TrivialAugmentWide, random erasing.
                       TrivialAugment (Mueller & Hutter, 2021) applies ONE randomly chosen operation per image
                       (rotation, shear, translation, colour, contrast, sharpness, posterize, solarize, equalize, ...)
                       at a random strength; it has no hyperparameters to tune, which suits a small dataset.

    Augmentation is applied to training images only. Evaluation (and the web app) resizes the short
    side to img_size * 256/224 (256 for 224 px, 512 for 448 px), then centre-crops img_size.
    mean / std must be the statistics the pretrained model expects (models.normalization).
    """
    if augment == "none":
        return eval_transform(img_size, mean, std), eval_transform(img_size, mean, std)
    if augment == "strong":
        train_tf = transforms.Compose([
            # down to 25% of the image area: much more variety in scale / position, but not so small that the
            # bird is usually cut out (the ImageNet default of 8% is too aggressive for fine-grained birds)
            transforms.RandomResizedCrop(img_size, scale=(0.25, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.TrivialAugmentWide(),                         # one random op, random magnitude
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.25),
        ])
        return train_tf, eval_transform(img_size, mean, std)
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),   # random zoom / crop
        transforms.RandomHorizontalFlip(),                           # birds face either way
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.25),                            # hides a random patch
    ])
    return train_tf, eval_transform(img_size, mean, std)


def eval_transform(img_size, mean=C.MEAN, std=C.STD):
    return transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224)),                # keeps the aspect ratio
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


# ---------------------------------------------------------------------------- dataset / loaders
class BirdDataset(Dataset):
    """Loads an image from disk and applies a transform. Returns (image tensor, label)."""

    def __init__(self, df, tf):
        # keep plain Python lists: fast to index from the DataLoader worker processes
        self.paths = df.path.tolist()
        self.labels = df.label.tolist()
        self.tf = tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        # images are read lazily, one at a time, so the whole dataset never has to fit in memory
        img = Image.open(self.paths[i]).convert("RGB")    # some JPEGs are greyscale
        return self.tf(img), self.labels[i]


def make_loader(df, img_size, batch_size, train, pin_memory=True, norm=(C.MEAN, C.STD), augment="basic"):
    """Training loaders shuffle, augment and drop the last incomplete batch; eval loaders do none.
    norm = (mean, std) of the model being trained / evaluated; augment = training augmentation preset."""
    train_tf, eval_tf = make_transforms(img_size, *norm, augment=augment)
    # shuffle: a new image order every epoch; drop_last: avoids a tiny last batch (bad for BatchNorm);
    # num_workers: images are decoded in parallel processes; pin_memory: faster copies to the GPU
    return DataLoader(BirdDataset(df, train_tf if train else eval_tf),
                      batch_size=batch_size, shuffle=train, drop_last=train,
                      num_workers=C.NUM_WORKERS, pin_memory=pin_memory)
