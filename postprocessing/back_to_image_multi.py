import cv2
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

SDSS_SOURCE_JPG = "/home/jovyan/data2/uniba/sdss_source_jpg/folder_1"
SDSS_LABELS = "/home/jovyan/data2/uniba/sdss_labels/folder_1"
SDSS_CROPPED_MASKS = "/home/jovyan/data2/uniba/sdss_cropped_masks/folder_1"
SDSS_WHOLE_MASKS = "/home/jovyan/data2/uniba/sdss_whole_masks/folder_1"


def paste_image(coords: str, img: np.ndarray, image_name: str, mask: np.ndarray, whole_masks_path):
    dh, dw = img.shape

    box = coords
    class_id, x_center, y_center, w, h, _ = box.strip().split()
    x_center, y_center, w, h = float(x_center), float(y_center), float(w), float(h)
    x_center = round(x_center * dw)
    y_center = round(y_center * dh)
    w = round(w * dw)
    h = round(h * dh)

    x = round(x_center - w / 2)
    y = round(y_center - h / 2)

    img[y : y + mask.shape[0], x : x + mask.shape[1]] = mask

    # Saving the image
    cv2.imwrite(os.path.join(whole_masks_path, image_name), img)


def create_whole_mask(
    mask, cropped_masks_path, whole_galaxies_path, labels_path, whole_masks_path
):
    missing_images = []
    missing_coords = []

    mask_image = cv2.imread(f"{cropped_masks_path}/{mask}")
    if mask_image is None:
        print(f"Error reading mask {mask}")
        return

    # Remove the 3rd dimension of the mask
    mask_image = mask_image[:, :, 0]

    mask_name, _, rank = mask.split(".")[0].split(" ")
    rank = int(rank)

    img = cv2.imread(f"{whole_galaxies_path}/{mask_name}.jpg", 0)
    if img is None:
        missing_images.append(mask_name)
        return

    # Paint the image black
    img *= 0

    try:
        with open(f"{labels_path}/{mask_name}.txt", "r") as label_file:
            coords = label_file.readlines()[rank]
            paste_image(coords, img, mask, mask_image, whole_masks_path)
    except FileNotFoundError:
        missing_coords.append(mask_name)

    if missing_images:
        print(f"Missing images: {missing_images}")
    if missing_coords:
        print(f"Missing coordinates: {missing_coords}")


def create_whole_masks(
    cropped_masks_path=SDSS_CROPPED_MASKS,
    whole_galaxies_path=SDSS_SOURCE_JPG,
    galaxies_labels_path=SDSS_LABELS,
    whole_masks_path=SDSS_WHOLE_MASKS,
):
    masks = os.listdir(cropped_masks_path)
    print("Processing images...")

    # Use ThreadPoolExecutor to process masks concurrently
    with ThreadPoolExecutor(max_workers=24) as executor:
        futures = [
            executor.submit(
                create_whole_mask,
                mask,
                cropped_masks_path,
                whole_galaxies_path,
                galaxies_labels_path,
                whole_masks_path,
            )
            for mask in masks
        ]

        # Optionally track progress
        for future in as_completed(futures):
            try:
                future.result()  # If an exception occurred, it will be raised here
            except Exception as e:
                print(f"An error occurred: {e}")


if __name__ == "__main__":
    create_whole_masks()
