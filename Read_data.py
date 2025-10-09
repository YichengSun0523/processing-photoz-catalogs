import os
import glob
import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
from tqdm import tqdm  
from collections import Counter

 
def read_one_col(folder_path, column_name,mask_func):
    '''return all valid data in an array filtered by mask function'''
    all_data = []
    fits_files = glob.glob(os.path.join(folder_path, "*.fits"))

    for file in tqdm(fits_files, desc="Reading FITS files"):
        try:
            with fits.open(file) as hdul:
                data = hdul[1].data
                table = np.array(data)
                mask = mask_func(table)
                filtered_data = table[column_name][mask]
                all_data.extend(filtered_data)
        except Exception as e:
            print(f"Failed to read {file}: {e}")

    return np.array(all_data)

def read_two_cols(folder_path, col1,col2,mask_func):
    '''return all valid data in an array'''
    all_data1 = []
    all_data2 = []
    fits_files = glob.glob(os.path.join(folder_path, "*.fits"))

    for file in tqdm(fits_files, desc="Reading FITS files"):
        try:
            with fits.open(file) as hdul:
                data = hdul[1].data
                table = np.array(data)
                mask = mask_func(table)
                all_data1.extend(table[col1][mask])
                all_data2.extend(table[col2][mask])
        except Exception as e:
            print(f"Failed to read {file}: {e}")

    return np.array(all_data1),np.array(all_data2)

def read_cols(folder_path, cols, mask_func):
    '''return all valid data for many columns (minimal change version)'''
    collectors = {c: [] for c in cols}
    fits_files = glob.glob(os.path.join(folder_path, "*.fits"))

    for file in tqdm(fits_files, desc="Reading FITS files"):
        try:
            with fits.open(file) as hdul:
                data = hdul[1].data
                table = np.array(data)
                mask = mask_func(table)
                for c in cols:
                    collectors[c].extend(table[c][mask])
        except Exception as e:
            print(f"Failed to read {file}: {e}")

    return tuple(np.array(collectors[c]) for c in cols)



mask_specz_exists= lambda tbl: tbl['spec_z'] != -10  
mask_specz_range= lambda tbl: (tbl['spec_z'] > 0) & (tbl['spec_z'] < 1) & (tbl['MASS_BEST'] > 7)
mask_photoz_range= lambda tbl: (tbl['photo_z'] > 0) & (tbl['photo_z'] < 1) & (tbl['MASS_BEST'] > 7)
mask_mass_range=lambda tbl: (tbl['MASS_BEST'] > 7) 
mask_photoz_error=lambda tbl: (tbl['photo_zerr'] < 0.1*(1+tbl['photo_z'])) & (tbl['spec_z'] != -10)
mask_everything = lambda tbl: np.ones(len(tbl), dtype=bool)



folder_path_zou="F:\photoz_mass_catalogs\phtoz_desidr9"
folder_path_vad="F:\VAD"

RA_zou,DEC_zou = read_two_cols(folder_path_zou, col1='RA',col2='DEC',mask_func=mask_everything)
RA_vad,DEC_vad = read_two_cols(folder_path_vad, col1='RA',col2='DEC',mask_func=mask_everything)
pos_zou=np.column_stack((RA_zou,DEC_zou))
pos_vad=np.column_stack((RA_vad,DEC_vad))

mass_zou,z_zou=read_two_cols(folder_path_zou,col1="MASS_BEST",col2="photo_z",mask_func=mask_everything)
mass_vad,z_vad=read_two_cols(folder_path_vad,col1="LOGM",col2="Z",mask_func=mask_everything)


np.savez("D:\zhuomian\Fall 2025 Berkeley\Astro Research\position_mass_z(zou).npz",mass=mass_zou,position=pos_zou,z=z_zou)
np.savez("D:\zhuomian\Fall 2025 Berkeley\Astro Research\position_mass_z(vad).npz",mass=mass_vad,position=pos_vad,z=z_vad)
