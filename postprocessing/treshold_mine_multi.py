import cv2
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# Path to input image and where masks will be saved
SDSS_CROPPED_GALAXIES = "/home/jovyan/data2/uniba/sdss_cropped_galaxies/folder_1"
SDSS_CROPPED_MASKS = "/home/jovyan/data2/uniba/sdss_cropped_masks/folder_1"


# Function to process a single image and save the mask
def create_mask(cropped_galaxy, cropped_galaxies_path, cropped_masks_path):
    cropped_galaxy_filename = cropped_galaxy.split(".")[0]
    image = cv2.imread(f"{cropped_galaxies_path}/{cropped_galaxy_filename}.jpg")

    if image is None:
        print(f"Error reading image {cropped_galaxy_filename}.jpg")
        return

    # Convert the image to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Otsu's thresholding
    thresh, im_bw = cv2.threshold(
        image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Create a binary mask using dilation and erosion
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.dilate(im_bw, kernel, iterations=2)
    mask = cv2.erode(mask, kernel, iterations=2)

    # Save the mask image
    cv2.imwrite(
        os.path.join(cropped_masks_path, f"{cropped_galaxy_filename}.jpg"), mask
    )


def create_masks(
    cropped_galaxies_path=SDSS_CROPPED_GALAXIES, cropped_masks_path=SDSS_CROPPED_MASKS
):
    cropped_galaxies = os.listdir(cropped_galaxies_path)

    # Use ThreadPoolExecutor to process images in parallel
    with ThreadPoolExecutor(max_workers=24) as executor:
        futures = [
            executor.submit(
                create_mask, cropped_galaxy, cropped_galaxies_path, cropped_masks_path
            )
            for cropped_galaxy in cropped_galaxies
        ]

        # Optionally, track progress as threads complete
        for future in as_completed(futures):
            try:
                future.result()  # This will raise any exceptions if they occur
            except Exception as e:
                print(f"An error occurred: {e}")


if __name__ == "__main__":
    create_masks()
