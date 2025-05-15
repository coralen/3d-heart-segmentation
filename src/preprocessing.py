import os
import numpy as np
import SimpleITK as sitk
import nibabel as nib
from scipy import ndimage
from scipy.ndimage import center_of_mass, shift
import glob
from tqdm import tqdm
import logging
import argparse
from pathlib import Path
import yaml
import json


class MedicalImagePipeline:
    def __init__(self, config_path=None, scan_metrics_path=None):
        self.preprocessor = MedicalImagePreprocessor(scan_metrics_path=scan_metrics_path)
        self.config = self.load_config(config_path)
        self.stages = self._build_pipeline()

    def load_config(self, config_path):
        """Load pipeline configuration from YAML file"""
        if config_path is None or not os.path.exists(config_path):
            print("No config file found. Using default settings.")
            return self._default_config()
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _default_config(self):
        """Default configuration if no config file is provided"""
        return {
            "flip_correction": True,
            "bias_correction": True,
            "denoising": True,
            "resampling": True,
            "intensity_normalization": True,
            "mask_alignment": True
        }

    def _build_pipeline(self):
        """Build the pipeline dynamically based on config"""
        stages = []
        if self.config['flip_correction']:
            stages.append(self.preprocessor.detect_and_correct_flip_handler)
        if self.config['bias_correction']:
            stages.append(self.preprocessor.log_bias_correction_handler)
        if self.config['denoising']:
            stages.append(self.preprocessor.denoise)
        if self.config['resampling']:
            stages.append(self.preprocessor.resample_handler)
        if self.config['intensity_normalization']:
            stages.append(self.preprocessor.normalize_intensity)
        if self.config['mask_alignment']:
            stages.append(self.preprocessor.align_mask)
        return stages

    def run(self, image, mask):
        """Run the configured pipeline on a single image and mask."""
        for stage in self.stages:
            # Check how many arguments the function expects
            if stage.__code__.co_argcount == 2:  # Takes only image
                image = stage(image)  # Run function with image only
            elif stage.__code__.co_argcount == 3:  # Takes image and mask
                image, mask = stage(image, mask)  # Run function with image and mask
            else:
                raise ValueError(f"Unexpected argument count for stage: {stage.__name__}")

        # Return both image and mask after processing
        return image, mask


class MedicalImagePreprocessor:
    def __init__(self, remove_noise_threshold=0.01, scan_metrics_path=None, min_spacing=(1.0, 1.0, 1.0)):
        """
        Initialize the medical image preprocessing pipeline.
        
        Args:
            remove_noise_threshold (float): Threshold for removing low-intensity noise
        """
        self.noise_threshold = remove_noise_threshold
        self.min_spacing = min_spacing 
        
        if scan_metrics_path:
            with open(scan_metrics_path, 'r') as f:
                metrics = json.load(f)
                spacings = [tuple(v["Voxel Spacing"]) for d in metrics.values() for v in d.values()]
                self.min_spacing = tuple(np.min(spacings, axis=0)) if spacings else self.min_spacing
        
    def detect_and_correct_flip_handler(self, image, mask):
        image_data = self.detect_and_correct_flip(image)
        mask_data = self.detect_and_correct_flip(mask)
        return image_data, mask_data
        
  
    def detect_and_correct_flip(self, img):
        """
        Detect and correct flipped orientation in medical images.
        
        Args:
            img: Image with header information (e.g., NIfTI)
            
        Returns:
            Corrected image data
        """
        qform = img.header.get_qform()
        sform = img.header.get_sform()
        img_data = img.get_fdata()

        if qform is not None and np.allclose(qform[:3, :3], np.diag([-1, -1, 1])):
            return np.flip(img_data, axis=(0, 1))  # Flip X and Y axes

        if sform is not None and np.allclose(sform[:3, :3], np.diag([-1, -1, 1])):
            return np.flip(img_data, axis=(0, 1))  # Flip X and Y axes

        return img_data

    def align_mask(self, image, mask):
        """
        Align a mask to an image using center of mass alignment.
        
        Args:
            image: Reference image data
            mask: Mask to be aligned
            
        Returns:
            Aligned mask
        """
        # Only consider non-zero values for center of mass calculation
        img_valid = image > 0
        mask_valid = mask > 0
        
        if np.sum(img_valid) == 0 or np.sum(mask_valid) == 0:
            return image, mask  # Skip alignment if either image is empty
            
        img_com = center_of_mass(image * img_valid)
        mask_com = center_of_mass(mask * mask_valid)

        if any(np.isnan(img_com)) or any(np.isnan(mask_com)):
            return image, mask  # Skip alignment if invalid center found

        shift_vector = np.array(img_com) - np.array(mask_com)
        aligned_mask = shift(mask, shift_vector, order=0)  # Use order=0 for nearest neighbor to preserve mask values
        
        # Ensure binary mask remains binary
        if np.array_equal(np.unique(mask), [0, 1]):
            aligned_mask = aligned_mask.astype(mask.dtype)
            
        return image, aligned_mask
        
    def log_bias_correction_handler(self, image):
        return self.log_bias_correction(image, epsilon=1e-6)

            
    def log_bias_correction(self, image, epsilon=1e-6):
        
        image = (image - image.min()) / (image.max() - image.min() + epsilon)
    
        corrected = np.log1p(image)
        corrected = (corrected - corrected.min()) / (corrected.max() - corrected.min()) * (image.max() - image.min()) + image.min()

        return corrected.astype(image.dtype)

    def normalize_intensity(self, image, mask):
        """
        Normalize image intensity to range [0,1].
        
        Args:
            image: Input image data
            
        Returns:
            Normalized image
        """
        # Only normalize if there's a range to normalize
        if np.max(image) > np.min(image):
            image = (image - np.min(image)) / (np.max(image) - np.min(image))
            
        #mask = (mask > 0).astype(np.uint8)
        return image, mask

    def denoise(self, image):
        """
        Apply denoising if the image has high standard deviation of intensity.
        
        Args:
            image: Input image data
            std_intensity: Standard deviation of intensity values
            
        Returns:
            Denoised image
        """

        std_intensity = np.std(image)
        
        if std_intensity > 100:
            sitk_image = sitk.GetImageFromArray(image)
            sitk_image = sitk.CurvatureFlow(image1=sitk_image, timeStep=0.125, numberOfIterations=2)
            return sitk.GetArrayFromImage(sitk_image)
        return image
        
    def resample_handler(self, image, mask):
        image_data = self.resample(image, new_spacing=self.min_spacing, reference_image=None, is_mask=False)
        mask_data = self.resample(mask, new_spacing=self.min_spacing, reference_image=image_data, is_mask=True)
        return image_data, mask_data

    def resample(self, image, new_spacing=(1.0, 1.0, 1.0), reference_image=None, is_mask=False):
        """
        Resample image to specified spacing or match a reference image.
        
        Args:
            image: Input image data
            new_spacing: Target voxel spacing (default: isotropic 1mm)
            reference_image: Optional reference image to match
            is_mask: Whether the image is a binary mask
            
        Returns:
            Resampled image
        """
        # Convert to SimpleITK format if needed
        if not isinstance(image, sitk.Image):
            sitk_image = sitk.GetImageFromArray(image)
        else:
            sitk_image = image
            
        # Set up the resampler
        resample = sitk.ResampleImageFilter()
        
        # Apply light smoothing before resampling (for images only, not masks)
        if not is_mask:
            smoothing = sitk.SmoothingRecursiveGaussianImageFilter()
            smoothing.SetSigma(0.5)
            sitk_image = smoothing.Execute(sitk_image)

        # Set output parameters based on reference image or specified spacing
        if reference_image is not None:
            if not isinstance(reference_image, sitk.Image):
                ref_sitk = sitk.GetImageFromArray(reference_image)
            else:
                ref_sitk = reference_image
                
            resample.SetReferenceImage(ref_sitk)
        else:
            # Calculate new size based on spacing change
            original_spacing = np.array(sitk_image.GetSpacing())
            original_size = np.array(sitk_image.GetSize())
            
            # Convert to 3D if input is 2D
            if len(original_spacing) < 3:
                original_spacing = np.append(original_spacing, 1.0)
                original_size = np.append(original_size, 1)
                new_spacing = new_spacing[:len(original_spacing)]
            
            new_size = np.round(original_size * (original_spacing / new_spacing)).astype(int)
            resample.SetOutputSpacing(new_spacing)
            resample.SetSize(new_size.tolist())
            resample.SetOutputOrigin(sitk_image.GetOrigin())
            resample.SetOutputDirection(sitk_image.GetDirection())

        # Set interpolator based on image type
        resample.SetInterpolator(sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear)

        # Execute resampling
        resampled_image = resample.Execute(sitk_image)
        
        # Convert back to numpy array
        return sitk.GetArrayFromImage(resampled_image)


def process_dataset(input_dir, output_dir, config_path="config.yml", scan_metrics_path=None):
    """Process all images and masks in a dataset using the pipeline."""
    pipeline = MedicalImagePipeline(config_path, scan_metrics_path)
    
    # Define input and output paths for images and masks
    image_dir = os.path.join(input_dir, "images")
    mask_dir = os.path.join(input_dir, "masks")

    output_image_dir = os.path.join(output_dir, "images")
    output_mask_dir = os.path.join(output_dir, "masks")

    # Create output directories if they don't exist
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    for filename in os.listdir(image_dir):
        if filename.endswith(".nii") or filename.endswith(".nii.gz"):
            image_path = os.path.join(image_dir, filename)
            mask_path = os.path.join(mask_dir, filename.replace(".nii", "_mask.nii"))
               
            # Load image and optional mask
            image = nib.load(image_path)
            mask = nib.load(mask_path) if os.path.exists(mask_path) else None

            if mask is not None:
                # Run the pipeline
                processed_image, processed_mask = pipeline.run(image, mask)

                # Save results
                output_image_path = os.path.join(output_image_dir, filename.replace(".nii", "_processed.nii"))
                nib.save(nib.Nifti1Image(processed_image, image.affine), output_image_path)
                output_mask_path = os.path.join(output_mask_dir, filename.replace(".nii", "_mask_processed.nii"))
                nib.save(nib.Nifti1Image(processed_mask, image.affine), output_mask_path)

def main():
    parser = argparse.ArgumentParser(description="Medical Image Processing Pipeline")
    parser.add_argument("--input_dir", type=str, required=True, 
                        help="Input directory containing /images and /masks folders")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for processed images and masks")
    parser.add_argument("--scan_metrics_path", type=str, required=True, 
                        help="Path to scan_metrics JSON file")
    parser.add_argument("--spacing_x", type=float, default=1.0,
                        help="Target spacing in x dimension (default: 1.0)")
    parser.add_argument("--spacing_y", type=float, default=1.0,
                        help="Target spacing in y dimension (default: 1.0)")
    parser.add_argument("--spacing_z", type=float, default=1.0,
                        help="Target spacing in z dimension (default: 1.0)")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process the dataset
    process_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        config_path="config.yml",
        scan_metrics_path=args.scan_metrics_path
    )
    
    print("Processing complete!")


if __name__ == "__main__":
    main()