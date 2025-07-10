import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from glob import glob
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd, ScaleIntensityRanged,
    Resized, SaveImaged, Compose
)
from monai.data import Dataset, DataLoader
from monai.utils import set_determinism

from config import RAW_DATA_DIR, PREPROCESSED_DIR


def main():
    set_determinism(seed=42)

    for dir in ["train", "val"]:
        input_dir = os.path.join(RAW_DATA_DIR, dir)
        output_dir = os.path.join(PREPROCESSED_DIR, dir)
        os.makedirs(output_dir, exist_ok=True)

        images = sorted(glob(os.path.join(input_dir, "images", "*")))
        masks = sorted(glob(os.path.join(input_dir, "masks", "*")))
        data_dicts = [{"img": img, "msk": msk} for img, msk in zip(images, masks)]
        
        # MONAI preprocessing pipeline
        preprocess_tf = Compose([
            LoadImaged(keys=["img", "msk"]),
            EnsureChannelFirstd(keys=["img", "msk"]),
            Orientationd(keys=["img", "msk"], axcodes="RAS"),
            Spacingd(keys=["img", "msk"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            ScaleIntensityRanged(
                keys=["img"], a_min=0, a_max=255,
                b_min=0.0, b_max=1.0, clip=True
            ),
            Resized(keys=["img", "msk"], spatial_size=(128, 128, 32), mode=("trilinear", "nearest")),
            SaveImaged(keys=["img"], output_dir=os.path.join(output_dir, "images"), separate_folder=False, output_postfix="pre"),
            SaveImaged(keys=["msk"], output_dir=os.path.join(output_dir, "masks"), separate_folder=False, output_postfix="pre")
        ])
        
        # Apply to all files
        dataset = Dataset(data=data_dicts, transform=preprocess_tf)
        loader = DataLoader(dataset, batch_size=1, num_workers=4)
    
        for _ in loader:
            pass 

if __name__ == "__main__":
    main()
