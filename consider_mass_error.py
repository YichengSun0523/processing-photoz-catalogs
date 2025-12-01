import numpy as np
import matplotlib.pyplot as plt

def best_est(est1,est2,err1,err2):
    '''return the best estimate based on two independent measurements'''
    w1 = 1.0 / (err1 ** 2)
    w2 = 1.0 / (err2 ** 2)
    

    est_best = (w1 * est1 + w2 * est2) / (w1 + w2)
    err_best = (1.0 / (w1 + w2)) ** 0.5
    
    return est_best, err_best


data_vad=np.load("D:\zhuomian\Fall 2025 Berkeley\Astro Research\position_mass_z_SigmaMass(well matched,vad).npz")
data_zou=np.load("D:\zhuomian\Fall 2025 Berkeley\Astro Research\position_mass_z_SigmaMass(well matched,zou).npz")

sigma_mass_vad=data_vad["SigmaMass"]
sigma_mass_zou=data_zou["SigmaMass"]

mass_vad=data_vad["mass"]
mass_zou=data_zou["mass"]

print(len(mass_vad))
print(len(mass_zou))

threshold=10
mass_combined=[]
mass_vad_filtered=[]
mass_zou_filtered=[]

for i in range(len(mass_vad)):
    if mass_zou[i]>threshold and mass_vad[i]>threshold and sigma_mass_vad[i]>0 and sigma_mass_zou[i]>0:
        mass_best=best_est(mass_vad[i],mass_zou[i],sigma_mass_vad[i],sigma_mass_zou[i])
        mass_combined.append(mass_best[0])
        mass_vad_filtered.append(mass_vad[i])
        mass_zou_filtered.append(mass_zou[i])
        
plt.hist(mass_combined,bins=100,density=True, histtype='step',color='green',label='combined')
plt.hist(mass_vad_filtered,bins=100,density=True, histtype='step',color='blue',label='value-added-catalog')
plt.hist(mass_zou_filtered,bins=100,density=True, histtype='step',color='red',label='Zou')

plt.yscale('log')
plt.xlabel("log(Mass)")
plt.ylabel("Number of Galaxies")
plt.title('Galaxy Mass Distribution (M*>10)')
plt.legend()
plt.show()
        