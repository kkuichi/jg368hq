#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Updated on Wed Feb 21 14:54:55 2024 with multithreading

@author: zc
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from astropy import units
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astroquery.ipac.ned import Ned
from astroquery.simbad import Simbad

# turn off warnings from astroquery
import warnings

warnings.filterwarnings("ignore")

EXTRACTED_FITS = "/home/jovyan/data2/uniba/sdss_extracted_fits/folder_1"
SDSS_LABELS = "/home/jovyan/data2/uniba/sdss_labels/folder_1"


@dataclass
class YoloOutput:
    x_center_norm: float
    y_center_norm: float
    confidence: float


@dataclass
class FitOutput:
    wcs: WCS
    fit_width: int
    fit_height: int


@dataclass
class GalaxyData:
    fit_name: str
    RA: float
    DEC: float
    Galaxy_Name_Simbad: str
    Galaxy_Name_NED: str
    Confidence: float


def extract_data_from_yolo(label_file: str, line_number: int) -> YoloOutput:
    with open(label_file, "r") as yolo_output_txt:
        yolo_output_list = yolo_output_txt.readlines()

    _, x_center_norm, y_center_norm, _, _, confidence = (
        yolo_output_list[line_number].strip().split()
    )
    return YoloOutput(
        x_center_norm=float(x_center_norm),
        y_center_norm=float(y_center_norm),
        confidence=float(confidence),
    )


def extract_data_from_fit(fit_file_path: str) -> FitOutput:
    with fits.open(fit_file_path) as hdu_list:
        image_shape = hdu_list[0].data.shape
        return FitOutput(
            wcs=WCS(hdu_list[0].header),
            fit_width=image_shape[1],
            fit_height=image_shape[0],
        )


def get_skycoord(fit_output: FitOutput, yolo_output: YoloOutput) -> SkyCoord:
    x_center = round(yolo_output.x_center_norm * fit_output.fit_width)
    y_center = round(fit_output.fit_height - (yolo_output.y_center_norm * fit_output.fit_height))

    central_pixel = np.array([[x_center, y_center]])
    world_coords = fit_output.wcs.pixel_to_world(central_pixel[:, 0], central_pixel[:, 1])
    ra_values = world_coords.ra.deg[0]
    dec_values = world_coords.dec.deg[0]

    return SkyCoord(ra=ra_values, dec=dec_values, unit="deg", frame="icrs")


def get_galaxy_name_simbad(ra_deg, dec_deg) -> str:
    coords = SkyCoord(ra=ra_deg, dec=dec_deg, unit="deg", frame="icrs", equinox="J2000.0")
    result_table = Simbad.query_region(coords, radius="0d0m6s")

    if result_table is None or len(result_table) <= 0:
        return "Galaxy not found"
    return result_table["MAIN_ID"][0]


def get_galaxy_name_ned(sky_coord: SkyCoord) -> str:
    result_table = Ned.query_region(sky_coord, radius=0.001 * units.deg, equinox="J2000.0")
    if result_table is None or len(result_table) <= 0:
        return "Galaxy not found"
    return result_table["Object Name"][0]


def get_galaxy_data(fit_file_path: str, label_file: str, line_number: int) -> GalaxyData:
    yolo_output = extract_data_from_yolo(label_file=label_file, line_number=line_number)
    fit_output = extract_data_from_fit(fit_file_path=fit_file_path)
    central_coords = get_skycoord(fit_output=fit_output, yolo_output=yolo_output)
    ra = np.round(central_coords.ra.deg, 3)
    dec = np.round(central_coords.dec.deg, 3)
    galaxy_name_1 = get_galaxy_name_simbad(ra, dec)
    galaxy_name_2 = get_galaxy_name_ned(central_coords)

    return GalaxyData(
        fit_name=fit_file_path.split("/")[-1],
        RA=ra,
        DEC=dec,
        Galaxy_Name_Simbad=galaxy_name_1,
        Galaxy_Name_NED=galaxy_name_2,
        Confidence=yolo_output.confidence,
    )


def get_galaxy(fit_filename: str, labels: str, fit_dir_path) -> GalaxyData:
    fit_filepath = os.path.join(fit_dir_path, fit_filename)
    label_file = os.path.join(labels, f"{fit_filename.split()[0]}.txt")
    line_number = int(fit_filename.split()[-1].split(".")[0])
    return get_galaxy_data(fit_filepath, label_file, line_number)


def get_galaxies(fit_dir_path: str = EXTRACTED_FITS, labels: str = SDSS_LABELS, ) -> pd.DataFrame:
    df_galaxy_data = pd.DataFrame(
        columns=["fit_name", "RA", "DEC", "Galaxy_Name_Simbad", "Galaxy_Name_NED", "Confidence"]
    )

    fit_filenames = os.listdir(fit_dir_path)

    # Use ThreadPoolExecutor to process multiple FITS files concurrently
    with ThreadPoolExecutor(max_workers=24) as executor:
        futures = [executor.submit(get_galaxy, fit_filename, labels, fit_dir_path) for fit_filename in fit_filenames]

        for future in as_completed(futures):
            try:
                galaxy_data = future.result()
                df_galaxy_data = df_galaxy_data.append(galaxy_data.__dict__, ignore_index=True)
            except Exception as e:
                print(f"Error processing FIT: {e}")

    return df_galaxy_data


if __name__ == "__main__":
    result_df = get_galaxies()
    print(f"Galaxies processed: {len(result_df)}")
    result_df.to_excel("/home/jovyan/data2/uniba/sdss_result_excel/galaxies_data_folder_1.xlsx", index=False)
