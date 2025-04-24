import os
import cv2
import numpy as np
from PIL import Image
from astropy.io import fits
from concurrent.futures import ThreadPoolExecutor, as_completed

# Define the paths to the FITS file and the mask in JPG format
SDSS_SOURCE_FITS = "/home/jovyan/data2/uniba/sdss_source_fits/folder_1"
SDSS_WHOLE_MASKS = "/home/jovyan/data2/uniba/sdss_whole_masks/folder_1"
EXTRACTED_FITS = "/home/jovyan/data2/uniba/sdss_extracted_fits/folder_1"


# Function to process a single FITS and mask pair
def extract_fit(
    whole_mask, source_fits_path, whole_masks_path, extracted_fits_path
):
    try:
        # Open the FITS file
        with fits.open(
            f"{source_fits_path}/{whole_mask.split(' ')[0]}.fits.bz2"
        ) as hdul:
            # Access the data array of the primary HDU (assuming it's the first extension)
            data = hdul[0].data

            # Open the mask image
            mask_image = Image.open(f"{whole_masks_path}/{whole_mask}")
            desired_width = data.shape[1]
            desired_height = data.shape[0]

            # Resize the mask image while maintaining the original shape
            resized_mask = mask_image.resize(
                (desired_width, desired_height), resample=Image.NEAREST
            )
            resized_mask.save("tmp.jpg")

            resized_mask = cv2.imread("tmp.jpg")
            resized_mask = resized_mask[:, :, 0]

            resized_mask = resized_mask[::-1]  # Flip the mask
            bin_mask = resized_mask < 50
            new_image = np.copy(data)
            new_image[bin_mask] = resized_mask[bin_mask]

            # Update the FITS data
            hdul_new = hdul
            hdul_new[0].data = new_image

            # Save the updated data to a new FITS file
            output_file_path = (
                f"{extracted_fits_path}/{whole_mask.split('.')[0]}.fits.bz2"
            )
            hdul_new.writeto(output_file_path, overwrite=True, output_verify="ignore")

    except Exception as e:
        print(f"Error processing {whole_mask}: {e}")


# Function to process FITS and masks in parallel
def extract_fits(
    source_fits_path=SDSS_SOURCE_FITS,
    whole_masks_path=SDSS_WHOLE_MASKS,
    extracted_fits_path=EXTRACTED_FITS,
):
    whole_masks = os.listdir(whole_masks_path)

    # Use ThreadPoolExecutor to process multiple files concurrently
    with ThreadPoolExecutor(max_workers=24) as executor:
        futures = [
            executor.submit(
                extract_fit,
                whole_mask,
                source_fits_path,
                whole_masks_path,
                extracted_fits_path,
            )
            for whole_mask in whole_masks
        ]

        # Optionally, track progress
        for future in as_completed(futures):
            try:
                future.result()  # This will raise any exceptions if they occurred
            except Exception as e:
                print(f"An error occurred: {e}")


if __name__ == "__main__":
    extract_fits()
