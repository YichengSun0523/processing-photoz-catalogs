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
pos_zou=np.column_stack((RA_zou,DEC_zou))
mass_zou_sup,mass_zou_inf = read_two_cols(folder_path_zou, col1='MASS_SUP',col2='MASS_INF',mask_func=mask_everything)
mass_zou_err=(mass_zou_sup-mass_zou_inf)/2

sigma_mass_vad=np.load("D:\zhuomian\Fall 2025 Berkeley\Astro Research\position_mass_z_SigmaMass(well matched,vad).npz")["SigmaMass"]
sigma_mass_zou=np.load("D:\zhuomian\Fall 2025 Berkeley\Astro Research\position_mass_z_SigmaMass(well matched,zou).npz")["SigmaMass"]

print(sigma_mass_vad)
print(sigma_mass_zou)

############ adding error of stellar mass to the .npz files ################## 
matched_zou=np.load("D:\zhuomian\Fall 2025 Berkeley\Astro Research\position_mass_z(well matched,zou).npz")

pos_zou_view = pos_zou.view(
    dtype=[('ra', pos_zou.dtype), ('dec', pos_zou.dtype)]
).reshape(-1)

matched_pos_zou = matched_zou['position']
matched_pos_view = matched_pos_zou.view(
    dtype=[('ra', matched_pos_zou.dtype), ('dec', matched_pos_zou.dtype)]
).reshape(-1)
order = np.argsort(pos_zou_view)
pos_zou_sorted = pos_zou_view[order]
idx_in_sorted = np.searchsorted(pos_zou_sorted, matched_pos_view)
idx_in_all = order[idx_in_sorted]
sigma_mass_zou = mass_zou_err[idx_in_all]

matched_zou_dict = dict(matched_zou)
matched_zou_dict['SigmaMass'] = sigma_mass_zou

np.savez("D:\zhuomian\Fall 2025 Berkeley\Astro Research\position_mass_z_SigmaMass(well matched,zou).npz",**matched_zou_dict)






RA_vad,DEC_vad = read_two_cols(folder_path_vad, col1='RA',col2='DEC',mask_func=mask_everything)
pos_vad=np.column_stack((RA_vad,DEC_vad))
mass_vad_err=read_one_col(folder_path_vad,column_name='LOGM_ERR',mask_func=mask_everything)
matched_vad=np.load("D:\zhuomian\Fall 2025 Berkeley\Astro Research\position_mass_z(well matched,vad).npz")

pos_vad_view = pos_vad.view(
    dtype=[('ra', pos_vad.dtype), ('dec', pos_vad.dtype)]
).reshape(-1)

matched_pos_vad = matched_vad['position']
matched_pos_view = matched_pos_vad.view(
    dtype=[('ra', matched_pos_vad.dtype), ('dec', matched_pos_vad.dtype)]
).reshape(-1)
order = np.argsort(pos_vad_view)
pos_vad_sorted = pos_vad_view[order]
idx_in_sorted = np.searchsorted(pos_vad_sorted, matched_pos_view)
idx_in_all = order[idx_in_sorted]
sigma_mass_vad = mass_vad_err[idx_in_all]


matched_vad_dict = dict(matched_vad)
matched_vad_dict['SigmaMass'] = sigma_mass_vad

np.savez("D:\zhuomian\Fall 2025 Berkeley\Astro Research\position_mass_z_SigmaMass(well matched,vad).npz",**matched_vad_dict)






#mass_zou,z_zou=read_two_cols(folder_path_zou,col1="MASS_BEST",col2="photo_z",mask_func=mask_everything)
#mass_vad,z_vad=read_two_cols(folder_path_vad,col1="LOGM",col2="Z",mask_func=mask_everything)
#np.savez("D:\zhuomian\Fall 2025 Berkeley\Astro Research\position_mass_z(zou).npz",mass=mass_zou,position=pos_zou,z=z_zou)
#np.savez("D:\zhuomian\Fall 2025 Berkeley\Astro Research\position_mass_z(vad).npz",mass=mass_vad,position=pos_vad,z=z_vad)
