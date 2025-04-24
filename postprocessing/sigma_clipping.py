import os
import numpy as np
import cv2
from astropy.stats import sigma_clip
from skimage.morphology import remove_small_objects


def refine_mask_with_sigma_clipping(mask, sigma=2, max_iters=5, min_size=10):
    """
    Applies sigma clipping to refine a binary segmentation mask, ensuring a black background.

    :param numpy.ndarray mask: Binary mask (non-mask = 0, mask = 255)
    :param float sigma: Sigma threshold for outlier detection.
    :param int max_iters: Maximum number of iterations for sigma clipping.
    :param int min_size: Minimum connected component size to keep.
    :return numpy.ndarray: Refined binary mask (0 = background, 255 = foreground).
    """
    # Ensure binary format (0 = background, 255 = foreground)
    mask = np.where(mask > 128, 255, 0).astype(np.float32)

    # Apply sigma clipping to foreground only
    clipped_mask = sigma_clip(mask, sigma=sigma, maxiters=max_iters, masked=True)

    # Create a new binary mask: 255 for foreground, 0 for background
    refined_mask = np.zeros_like(mask, dtype=np.uint8)
    refined_mask[~clipped_mask.mask] = 255  # Restore only valid foreground pixels

    # Remove small noise components
    refined_mask = remove_small_objects(refined_mask.astype(bool), min_size=min_size).astype(np.uint8) * 255

    # Invert colors (flip black and white)
    refined_mask = cv2.bitwise_not(refined_mask)

    return refined_mask


def process_masks_in_folder(input_folder, output_folder, sigma=2, max_iters=5, min_size=10):
    """
    Iterates through a folder, refines all mask images using sigma clipping, and saves them as JPG.

    :param str input_folder: Path to the folder containing PNG masks.
    :param str output_folder: Path to the folder where refined masks will be saved as JPG.
    :param float sigma: Sigma threshold for sigma clipping.
    :param int max_iters: Maximum number of iterations for sigma clipping.
    :param int min_size: Minimum connected component size to keep.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get all PNG mask filenames
    mask_filenames = sorted([
        f for f in os.listdir(input_folder) if f.lower().endswith('.png')
    ])

    for mask_filename in mask_filenames:
        input_path = os.path.join(input_folder, mask_filename)

        # Change extension to .jpg
        output_filename = os.path.splitext(mask_filename)[0] + ".jpg"
        output_path = os.path.join(output_folder, output_filename)

        # Read mask in grayscale
        mask = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            print(f"Warning: Could not read {input_path}")
            continue

        # Apply sigma clipping to refine mask
        refined_mask = refine_mask_with_sigma_clipping(mask, sigma=sigma, max_iters=max_iters, min_size=min_size)

        # Save the refined mask as JPG
        cv2.imwrite(output_path, refined_mask, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        #print(f"Saved refined mask: {output_path}")
    print('Done')

if __name__ == "__main__":
    input_folder = "../scss-net/scss-net/data/galaxies_png/predicted_masks_original_shape"  # Folder with PNG masks
    output_folder = "../data/new_data_subset/sigma_masks/sigma_2_0/masks"  # Output folder for JPG masks
    process_masks_in_folder(input_folder, output_folder, sigma=2, max_iters=5)