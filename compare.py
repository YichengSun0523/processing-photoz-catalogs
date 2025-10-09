import matplotlib.pyplot as plt
import numpy as np
import re
import glob
import os
from scipy.stats import linregress, pearsonr

def compare(m1,m2,title,save_path):
    plt.hist2d(m1, m2, bins=100,norm='log', cmap='Greys')
    plt.xlabel("mass in vad")
    plt.ylabel("mass in zou")
    plt.title(title)
    

    slope, intercept, r_value, p_value, std_err = linregress(m2, m1)
    delta = np.median(m1 - m2)
    nmad = 1.4826 * np.median(np.abs((m1 - m2)))
    pearson_r, pearson_p = pearsonr(m2, m1)
    
    
    ax = plt.gca()
    x0, x1 = ax.get_xlim()
    x = np.linspace(x0, x1, 200)

    # y = x
    ax.plot(x, x, linestyle='--', linewidth=1, label='y = x')
    # linear fit
    ax.plot(x, slope*x + intercept, linewidth=1.5, label=f'fit: y = {slope:.3f}x + {intercept:.3f}')

    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Linear regression: y = {slope:.4f} * x + {intercept:.4f}")
    print(f"Median difference (Δ): {delta:.4f}")
    print(f"NMAD: {nmad:.4f}")
    print(f"Pearson r: {pearson_r:.4f}")


m1=np.load("D:\zhuomian\Fall 2025 Berkeley\Astro Research\position_mass_z(well matched,vad).npz")['mass']
m2=np.load("D:\zhuomian\Fall 2025 Berkeley\Astro Research\position_mass_z(well matched,zou).npz")['mass']

print(len(m1))
print(len(m2))
compare(m1,m2,title=f"All data,(n={len(m1)})",save_path=f"D:\zhuomian\Fall 2025 Berkeley\Astro Research\divided into redshift bins\plot\All bins.png")

# —— 配对并逐对调用 compare ——
root = "D:\zhuomian\Fall 2025 Berkeley\Astro Research\divided into redshift bins\data"
pat  = re.compile(r"z_bin=([0-9.]+)-([0-9.]+)")

# 收集文件：bin -> 路径
vad_map, zou_map = {}, {}

for p in glob.glob(os.path.join(root, "mass_vad(z_bin=*.npy")):
    m = pat.search(os.path.basename(p))
    if m:
        key = (float(m.group(1)), float(m.group(2)))
        vad_map[key] = p

for p in glob.glob(os.path.join(root, "mass_zou(z_bin=*.npy")):
    m = pat.search(os.path.basename(p))
    if m:
        key = (float(m.group(1)), float(m.group(2)))
        zou_map[key] = p


# 取公共的 z-bin，并按下界排序
common_bins = sorted(set(vad_map) & set(zou_map), key=lambda k: k[0])

for (lo, hi) in common_bins:
    p_vad = vad_map[(lo, hi)]
    p_zou = zou_map[(lo, hi)]

    m_vad = np.load(p_vad, allow_pickle=True)
    m_zou = np.load(p_zou, allow_pickle=True)

    # filter and ravel into 1D
    m_vad = np.ravel(m_vad)
    m_zou = np.ravel(m_zou)
    mask = (m_vad > 10) & (m_zou > 10)
    m_vad = m_vad[mask]
    m_zou = m_zou[mask]

    # 若两侧长度不一致，先对齐到相同长度（按前 n 个）——
    # 这假设两侧已按相同对象顺序对齐；若不是，请先做对象级匹配。
    n = min(len(m_vad), len(m_zou))
    if n == 0:
        print(f"skip z-bin {lo}-{hi}: empty after filtering")
        continue
    if len(m_vad) != len(m_zou):
        print(f"warning: z-bin {lo}-{hi} size mismatch (vad={len(m_vad)}, zou={len(m_zou)}); using first {n}")

    print(f"\n=== z-bin {lo}-{hi} (n={n}) ===")
    compare(m_vad[:n], m_zou[:n],title=f"z-bin {lo}-{hi},(n={n})",save_path=f"D:\zhuomian\Fall 2025 Berkeley\Astro Research\divided into redshift bins\plot\z-bin {lo}-{hi}.png")