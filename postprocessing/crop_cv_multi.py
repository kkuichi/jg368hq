import cv2
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

SDSS_SOURCE_JPG = "/home/jovyan/data2/uniba/sdss_source_jpg/folder_1"
SDSS_LABELS = "/home/jovyan/data2/uniba/sdss_labels/folder_1"
SDSS_CROPPED_GALAXIES = "/home/jovyan/data2/uniba/sdss_cropped_galaxies/folder_1"


def crop_image(coords: str, img: np.ndarray, image_name: str, cropped_galaxies_path: str):
    dh, dw, _ = img.shape

    box = coords
    _, x_center, y_center, w, h, _ = box.strip().split()
    x_center, y_center, w, h = float(x_center), float(y_center), float(w), float(h)
    x_center = round(x_center * dw)
    y_center = round(y_center * dh)
    w = round(w * dw)
    h = round(h * dh)
    x = round(x_center - w / 2)
    y = round(y_center - h / 2)

    cropped_image = img[y:y + h, x:x + w]

    # Save the image
    cv2.imwrite(os.path.join(cropped_galaxies_path, image_name), cropped_image)


def crop_galaxies(label_file: str, labels_path: str, image_path: str, destination: str):
    label_name = label_file.split('.')[0]

    galaxy_img = cv2.imread(f"{image_path}/{label_name}.jpg")

    if galaxy_img is None:
        print(f"Error reading image {label_name}.jpg")
        return

    with open(f"{labels_path}/{label_file}", 'r') as file:
        labels = file.readlines()

    for i, label in enumerate(labels):
        crop_image(label, galaxy_img, f"{label_name} - {i}.jpg", destination)


def crop_and_save(labels_path=SDSS_LABELS, galaxies_jpg_path=SDSS_SOURCE_JPG, destination=SDSS_CROPPED_GALAXIES):
    label_files = os.listdir(labels_path)
    print("Processing images...")

    # Using ThreadPoolExecutor for multithreading
    with ThreadPoolExecutor(max_workers=24) as executor:
        # Submit tasks to the thread pool
        futures = [executor.submit(crop_galaxies, label_file, labels_path, galaxies_jpg_path, destination) for label_file in label_files]

        # Optionally, you can track progress using as_completed
        for future in as_completed(futures):
            try:
                future.result()  # If any exception occurs, it will be raised here
            except Exception as e:
                print(f"An error occurred: {e}")


if __name__ == "__main__":
    crop_and_save()
