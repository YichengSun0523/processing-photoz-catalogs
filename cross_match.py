import os
import glob
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
import time
from tqdm import tqdm  
from scipy.stats import linregress, pearsonr

'''
start_time=time.time()

data_zou = np.load("D:\zhuomian\Fall 2025 Berkeley\Astro Research\position_mass_z(zou).npz",allow_pickle=True)
data_vad = np.load("D:\zhuomian\Fall 2025 Berkeley\Astro Research\position_mass_z(vad).npz",allow_pickle=True)

z_zou=data_zou['z']
z_vad=data_vad['z']
position_zou=data_zou['position']
position_vad=data_vad['position']
mass_zou=data_zou['mass']
mass_vad=data_vad['mass']


# 1) 统一 ID 类型，避免因为 dtype 不一致导致匹配失败
ID_zou = ID_zou.astype(np.int64, copy=False)
ID_vad = ID_vad.astype(np.int64, copy=False)

# 2) 建立参考映射（可先去重）
# 若 ID_vad 有重复，按需要选择 keep='first'/'last' 或聚合方式
map_vad = pd.Series(mass_vad, index=ID_vad)
map_vad = map_vad[~map_vad.index.duplicated(keep='first')]

# 3) 分块对齐：对每块的 ID_zou 用 reindex 取出 mass_vad
chunk_size = 200_000  # 可按内存调整
rows = []
for start in range(0, len(ID_zou), chunk_size):
    end = min(start + chunk_size, len(ID_zou))
    ids_chunk = ID_zou[start:end]
    mass_zou_chunk = mass_zou[start:end]

    # 对齐：索引为 ids_chunk 的顺序，没匹配到的是 NaN
    mass_vad_chunk = map_vad.reindex(ids_chunk).to_numpy()

    # 只保留成功匹配的行（如果你想保留未匹配也行，就别 drop）
    mask = ~np.isnan(mass_vad_chunk)
    rows.append(pd.DataFrame({
        "ID": ids_chunk[mask],
        "mass_zou": mass_zou_chunk[mask],
        "mass_vad": mass_vad_chunk[mask]
    }))

matched_df = pd.concat(rows, ignore_index=True)
matched_df.to_csv("D:\zhuomian\Fall 2025 Berkeley\Astro Research\ID_match_result.csv", index=False)

end_time=time.time()
print("运行时间: {:.4f} 秒".format(end_time - start_time))
'''
'''
x1, y1 = position_zou[:,0], position_zou[:,1]
x2, y2 = position_vad[:,0], position_vad[:,1]

bins = 200
xmin = min(x1.min(), x2.min())
xmax = max(x1.max(), x2.max())
ymin = min(y1.min(), y2.min())
ymax = max(y1.max(), y2.max())

H1, xedges, yedges = np.histogram2d(x1, y1, bins=bins, range=[[xmin,xmax],[ymin,ymax]])
H2, _, _           = np.histogram2d(x2, y2, bins=[xedges, yedges])

plt.figure(figsize=(6,5))

plt.imshow(H1.T, origin="lower", extent=[xmin,xmax,ymin,ymax], 
           cmap="Reds", alpha=0.6, aspect="auto")

plt.imshow(H2.T, origin="lower", extent=[xmin,xmax,ymin,ymax], 
           cmap="Blues", alpha=0.6, aspect="auto")

plt.colorbar(label="Counts (overlap shows purple)")
plt.xlabel("RA")
plt.ylabel("DEC")
plt.title("Red:zou Blue:vad")
plt.tight_layout()
plt.show()
'''
'''
start_time=time.time()

ra_z, dec_z = position_zou[:, 0], position_zou[:, 1]
ra_v, dec_v = position_vad[:, 0], position_vad[:, 1]

cat_v = SkyCoord(ra_v * u.deg, dec_v * u.deg, frame='icrs')

ZOU_CHUNK = 300000  # 按内存调小/调大

m = len(cat_v)
best_sep2d = np.full(m, np.inf) * u.deg
best_idx   = np.full(m, -1, dtype=np.int64)

Nz = len(ra_z)
for z_start in range(0, Nz, ZOU_CHUNK):
    z_end = min(z_start + ZOU_CHUNK, Nz)

    # 当前块
    ra_z_chunk  = ra_z[z_start:z_end]
    dec_z_chunk = dec_z[z_start:z_end]
    cat_z = SkyCoord(ra_z_chunk * u.deg, dec_z_chunk * u.deg, frame='icrs')

    # 在该块内做最近邻匹配
    idx_chunk, sep2d_chunk, _ = cat_v.match_to_catalog_sky(cat_z)

    # 与目前最优解比较，保留更小角距离
    better = sep2d_chunk < best_sep2d
    if np.any(better):
        best_sep2d[better] = sep2d_chunk[better]
        best_idx[better]   = idx_chunk[better] + z_start  # 映射到全局 ZOU 索引

idx    = best_idx
sep2d  = best_sep2d
dist_arcsec = sep2d.to(u.arcsec).value

mass_vad_matched = mass_vad
mass_zou_matched = mass_zou[idx]
print(f"mass_vad_matched={mass_vad_matched},length={len(mass_vad_matched)}")
print(f"mass_zou_matched={mass_zou_matched},length={len(mass_zou_matched)}")

z_vad_matched = z_vad
z_zou_matched = z_zou[idx]
print(f"z_vad_matched={z_vad_matched},length={len(z_vad_matched)}")
print(f"z_zou_matched={z_zou_matched},length={len(z_zou_matched)}")

position_vad_matched = position_vad
position_zou_matched = position_zou[idx]
print(f"position_vad_matched={position_vad_matched},length={len(position_vad_matched)}")
print(f"position_zou_matched={position_zou_matched},length={len(position_zou_matched)}")

np.savez("D:\zhuomian\Fall 2025 Berkeley\Astro Research\position_mass_z(matched,vad).npz", position=position_vad_matched,mass=mass_vad_matched,z=z_vad_matched)
np.savez("D:\zhuomian\Fall 2025 Berkeley\Astro Research\position_mass_z(matched,zou).npz", position=position_zou_matched,mass=mass_zou_matched,z=z_zou_matched)
np.save("D:\zhuomian\Fall 2025 Berkeley\Astro Research\dist_arcsec(second test).npy",dist_arcsec )

end_time=time.time()

print(f'mass_vad_matched={mass_vad_matched}')
print(f'mass_zou_matched={mass_zou_matched}')
print("角距离(arcsec):", dist_arcsec)
print("运行时间: {:.4f} 秒".format(end_time - start_time))
'''


data_vad_matched=np.load("D:\zhuomian\Fall 2025 Berkeley\Astro Research\position_mass_z(matched,vad).npz")
data_zou_matched=np.load("D:\zhuomian\Fall 2025 Berkeley\Astro Research\position_mass_z(matched,zou).npz")
dist_arcsec =np.load("D:\zhuomian\Fall 2025 Berkeley\Astro Research\dist_arcsec(second test).npy")

mass_vad_matched=data_vad_matched["mass"]
mass_zou_matched=data_zou_matched["mass"]
z_vad_matched=data_vad_matched["z"]
z_zou_matched=data_zou_matched["z"]
position_vad_matched=data_vad_matched["position"]
position_zou_matched=data_zou_matched["position"]

mask=(dist_arcsec<0.01) & (mass_vad_matched>0) & (mass_zou_matched>0)

mass_vad_well_matched=mass_vad_matched[mask]
mass_zou_well_matched=mass_zou_matched[mask]
z_vad_well_matched=z_vad_matched[mask]
z_zou_well_matched=z_zou_matched[mask]
position_vad_well_matched=position_vad_matched[mask]
position_zou_well_matched=position_zou_matched[mask]
dist_arcsec_well_matched=dist_arcsec[mask]

np.savez("D:\zhuomian\Fall 2025 Berkeley\Astro Research\position_mass_z(well matched,vad).npz", position=position_vad_well_matched,mass=mass_vad_well_matched,z=z_vad_well_matched)
np.savez("D:\zhuomian\Fall 2025 Berkeley\Astro Research\position_mass_z(well matched,zou).npz", position=position_zou_well_matched,mass=mass_zou_well_matched,z=z_zou_well_matched)
np.save("D:\zhuomian\Fall 2025 Berkeley\Astro Research\dist_arcsec(second test, well matched).npy",dist_arcsec_well_matched )


print(f"Number of objects: {len(mass_vad_well_matched)}")
'''
values, counts = np.unique(ID_vad_well_matched-ID_zou_well_matched, return_counts=True)
mask = counts > 0
filtered_values = values[mask]
filtered_counts = counts[mask]
print("唯一值:", filtered_values)
print("次数:", filtered_counts)
'''
plt.hist2d(mass_vad_well_matched, mass_zou_well_matched, bins=100,norm='log', cmap='Greys')
plt.xlabel("mass in vad")
plt.ylabel("mass in zou")
plt.title("Cross Compare For Stellar Mass in zou and vad")
plt.show()

# 1. 线性回归 (y = a*x + b)
slope, intercept, r_value, p_value, std_err = linregress(mass_zou_well_matched, mass_vad_well_matched)

# 2. median difference
delta = np.median(mass_vad_well_matched - mass_zou_well_matched)

# 3. NMAD
nmad = 1.4826 * np.median(np.abs((mass_vad_well_matched - mass_zou_well_matched)))

# 4. Pearson coefficient
pearson_r, pearson_p = pearsonr(mass_zou_well_matched, mass_vad_well_matched)

print(f"Linear regression: y = {slope:.4f} * x + {intercept:.4f}")
print(f"Median difference (Δ): {delta:.4f}")
print(f"NMAD: {nmad:.4f}")
print(f"Pearson r: {pearson_r:.4f}")