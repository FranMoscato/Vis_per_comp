from __future__ import annotations

import argparse
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse

import numpy as np
import requests
import SimpleITK as sitk
import torch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.networks.nets import UNet
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    EnsureChannelFirstd,
    KeepLargestConnectedComponent,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandSpatialCropd,
    ToTensord,
)
from skimage.morphology import remove_small_objects
from skimage.segmentation import clear_border
from sklearn.model_selection import train_test_split
from tqdm import tqdm


PENGWIN_RECORD_ID = "10927452"
PENGWIN_IMAGES_URL = (
    "https://zenodo.org/records/10927452/files/"
    "PENGWIN_CT_train_images_part1.zip?download=1"
)
PENGWIN_LABELS_URL = (
    "https://zenodo.org/records/10927452/files/"
    "PENGWIN_CT_train_labels.zip?download=1"
)


@dataclass(frozen=True)
class DataConfig:
    root: Path = Path("Data")
    min_hu: float = 50.0
    max_hu: float = 1000.0

    @property
    def images_raw_dir(self) -> Path:
        return self.root / "PENGWIN_CT_train_images_part1"

    @property
    def labels_raw_dir(self) -> Path:
        return self.root / "PENGWIN_CT_train_labels"

    @property
    def images_normalized_dir(self) -> Path:
        return self.root / "PENGWIN_CT_train_images_normalized"

    @property
    def labels_binary_dir(self) -> Path:
        return self.root / "PENGWIN_CT_train_labels_binary"


def validate_zenodo_url(url: str) -> None:
    parsed = urlparse(url)
    expected_prefix = f"/records/{PENGWIN_RECORD_ID}/files/"
    if parsed.scheme != "https":
        raise ValueError("Only https downloads are allowed")
    if parsed.hostname != "zenodo.org":
        raise ValueError("Only zenodo.org downloads are allowed")
    if not parsed.path.startswith(expected_prefix):
        raise ValueError("URL does not match the expected PENGWIN Zenodo record")


def safe_extract_zip(zip_path: Path, target_folder: Path) -> None:
    target_root = target_folder.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            destination = (target_folder / member.filename).resolve()
            if not destination.is_relative_to(target_root):
                raise RuntimeError(f"Unsafe path inside zip: {member.filename}")
        zf.extractall(target_folder)


@contextmanager
def open_validated_download(url: str, max_redirects: int = 3):
    current_url = url
    response = None
    try:
        for _ in range(max_redirects + 1):
            validate_zenodo_url(current_url)
            response = requests.get(
                current_url,
                stream=True,
                timeout=(10, 60),
                allow_redirects=False,
            )
            if response.is_redirect:
                location = response.headers.get("location")
                response.close()
                response = None
                if not location:
                    raise RuntimeError("Download redirect without Location header")
                current_url = urljoin(current_url, location)
                continue

            response.raise_for_status()
            yield response
            return

        raise RuntimeError("Too many redirects while downloading dataset")
    finally:
        if response is not None:
            response.close()


def ensure_dataset(url: str, target_folder: Path, expected_ext: str = ".mha") -> None:
    target_folder.mkdir(parents=True, exist_ok=True)
    if any(target_folder.rglob(f"*{expected_ext}")):
        print(f"[OK] {target_folder.name} already contains data")
        return

    validate_zenodo_url(url)
    zip_path = target_folder.parent / f"{target_folder.name}.zip"
    print(f"Downloading {url}")
    with open_validated_download(url) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(zip_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc="Downloading"
        ) as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    print(f"Extracting into {target_folder}")
    safe_extract_zip(zip_path, target_folder)
    zip_path.unlink()
    print(f"[OK] Data ready at {target_folder}")


def binarize_mha(input_path: Path, output_path: Path) -> None:
    itk_img = sitk.ReadImage(str(input_path))
    img_array = sitk.GetArrayFromImage(itk_img)
    binary_array = (img_array > 0).astype(np.uint8)
    binary_img = sitk.GetImageFromArray(binary_array)
    binary_img.CopyInformation(itk_img)
    sitk.WriteImage(binary_img, str(output_path))


def normalize_ct(input_path: Path, output_path: Path, min_hu: float, max_hu: float) -> None:
    itk_img = sitk.ReadImage(str(input_path))
    img_array = sitk.GetArrayFromImage(itk_img).astype(np.float32)
    img_array = np.clip(img_array, min_hu, max_hu)
    img_array = (img_array - min_hu) / (max_hu - min_hu)
    processed_img = sitk.GetImageFromArray(img_array)
    processed_img.CopyInformation(itk_img)
    sitk.WriteImage(processed_img, str(output_path))


def preprocess_data(config: DataConfig) -> None:
    ensure_dataset(PENGWIN_IMAGES_URL, config.images_raw_dir)
    ensure_dataset(PENGWIN_LABELS_URL, config.labels_raw_dir)

    config.images_normalized_dir.mkdir(parents=True, exist_ok=True)
    config.labels_binary_dir.mkdir(parents=True, exist_ok=True)

    for label_path in tqdm(sorted(config.labels_raw_dir.glob("*.mha")), desc="Labels"):
        target_path = config.labels_binary_dir / label_path.name
        if not target_path.exists():
            binarize_mha(label_path, target_path)

    for image_path in tqdm(sorted(config.images_raw_dir.glob("*.mha")), desc="Images"):
        target_path = config.images_normalized_dir / image_path.name
        if not target_path.exists():
            normalize_ct(image_path, target_path, config.min_hu, config.max_hu)


def build_data_dicts(images_dir: Path, labels_dir: Path) -> list[dict[str, str]]:
    images = {path.stem: path for path in sorted(images_dir.glob("*.mha"))}
    labels = {path.stem: path for path in sorted(labels_dir.glob("*.mha"))}
    missing_labels = sorted(set(images) - set(labels))
    missing_images = sorted(set(labels) - set(images))

    if missing_labels or missing_images:
        raise ValueError(
            "Image/label mismatch. "
            f"missing_labels={missing_labels[:5]}, missing_images={missing_images[:5]}"
        )
    if not images:
        raise FileNotFoundError(f"No .mha images found in {images_dir}")

    return [
        {"image": str(images[stem]), "label": str(labels[stem])}
        for stem in sorted(images)
    ]


def split_data(
    data_dicts: list[dict[str, str]],
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    return train_test_split(data_dicts, test_size=test_size, random_state=random_state)


def build_transforms(variant: str = "context"):
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"], image_only=True),
            EnsureChannelFirstd(keys=["image", "label"]),
            ToTensord(keys=["image", "label"]),
        ]
    )

    if variant == "context":
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"], image_only=True),
                EnsureChannelFirstd(keys=["image", "label"]),
                RandSpatialCropd(
                    keys=["image", "label"],
                    roi_size=[128, 128, 128],
                    random_center=True,
                    random_size=False,
                ),
                ToTensord(keys=["image", "label"]),
            ]
        )
    elif variant == "foreground":
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"], image_only=True),
                EnsureChannelFirstd(keys=["image", "label"]),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(128, 128, 128),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=0,
                ),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                ToTensord(keys=["image", "label"]),
            ]
        )
    else:
        raise ValueError("variant must be 'context' or 'foreground'")

    return train_transforms, val_transforms


def build_model(device: torch.device) -> UNet:
    return UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)


def build_loss() -> DiceCELoss:
    return DiceCELoss(sigmoid=True)


def infer_volume(model: UNet, inputs: torch.Tensor, roi_size=(128, 128, 128)) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return sliding_window_inference(
            inputs=inputs,
            roi_size=roi_size,
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
        )


def postprocess_prediction(
    logits: torch.Tensor,
    threshold: float = 0.9,
    min_size: int = 3000,
    remove_border: bool = False,
) -> torch.Tensor:
    post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=threshold)])
    keep_components = KeepLargestConnectedComponent(
        applied_labels=[1],
        is_onehot=False,
        connectivity=3,
        num_components=1,
    )

    pred_raw = post_pred(logits)
    pred_np = pred_raw[0, 0].cpu().numpy().astype(bool)
    if remove_border:
        pred_np = clear_border(pred_np)
    pred_np = remove_small_objects(pred_np, min_size=min_size, connectivity=3)
    pred_clean = torch.from_numpy(pred_np.astype(np.uint8)).unsqueeze(0).unsqueeze(0)
    pred_clean = pred_clean.to(logits.device)
    return keep_components(pred_clean)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PENGWIN preprocessing utilities")
    parser.add_argument(
        "command",
        choices=["preprocess", "check-pairs"],
        help="Operation to run",
    )
    parser.add_argument(
        "--data-root",
        default="Data",
        type=Path,
        help="Directory containing or receiving PENGWIN data",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DataConfig(root=args.data_root)
    if args.command == "preprocess":
        preprocess_data(config)
    elif args.command == "check-pairs":
        data_dicts = build_data_dicts(config.images_normalized_dir, config.labels_binary_dir)
        print(f"[OK] Found {len(data_dicts)} matched image/label pairs")


if __name__ == "__main__":
    main()
