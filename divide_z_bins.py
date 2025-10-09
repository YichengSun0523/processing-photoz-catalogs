import numpy as np
import matplotlib.pyplot as plt
import os

def divide_data(x, y, bins):
    '''divide y according to the value of x'''
    bin_indices = np.digitize(x, bins) - 1 
    divided_y = [[] for _ in range(len(bins)-1)]
    for i, idx in enumerate(bin_indices):
        if 0 <= idx < len(bins)-1:
            divided_y[idx].append(y[i])
    return divided_y

def save_divided_npy(divided, out_prefix):
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    for i, bucket in enumerate(divided):
        arr = np.array(bucket)
        z_lo = i * 0.1
        z_hi = i * 0.1 + 0.1
        fname = f"{out_prefix}(z_bin={z_lo:.1f}-{z_hi:.1f}).npy"
        np.save(fname, arr)


data_zou = np.load("D:\zhuomian\Fall 2025 Berkeley\Astro Research\position_mass_z(well matched,zou).npz")
data_vad = np.load("D:\zhuomian\Fall 2025 Berkeley\Astro Research\position_mass_z(well matched,vad).npz")

mass_zou=data_zou["mass"]
z_zou=data_zou["z"]
position_zou=data_zou["position"]
mass_vad=data_vad["mass"]
z_vad=data_vad["z"]
position_vad=data_vad["position"]


ten_intervals = np.linspace(0, 1, 11)

divided_mass_zou = divide_data(z_zou, mass_zou, ten_intervals)
divided_mass_vad = divide_data(z_zou, mass_vad, ten_intervals)

divided_position_zou = divide_data(z_zou, position_zou, ten_intervals)
divided_position_vad = divide_data(z_zou, position_vad, ten_intervals)

save_divided_npy(divided_mass_zou,      "D:\zhuomian\Fall 2025 Berkeley\Astro Research\divided into redshift bins\mass_zou")
save_divided_npy(divided_mass_vad,      "D:\zhuomian\Fall 2025 Berkeley\Astro Research\divided into redshift bins\mass_vad")
save_divided_npy(divided_position_zou,  "D:\zhuomian\Fall 2025 Berkeley\Astro Research\divided into redshift bins\pos_zou")
save_divided_npy(divided_position_vad,  "D:\zhuomian\Fall 2025 Berkeley\Astro Research\divided into redshift bins\pos_vad")



# 绘制 redshift 分布图
plt.figure(figsize=(10, 6))
plt.hist2d(z_vad, z_zou, bins=100, range=[[0,1],[0,1]],norm='log', cmap='Greys')
#plt.colorbar()
#plt.hist([corresponding_specz,corresponding_photoz], bins=10, color=["red","blue"],label=["spec","photo"], alpha=0.7,density=False,histtype='step')
#plt.yscale('log')
plt.xlabel("z", fontsize=12)
plt.ylabel("Number of galaxies", fontsize=12)
plt.title("Redshift Distribution for 100 heaviest galaxies in 0<z<0.1 ", fontsize=14)
#plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()
