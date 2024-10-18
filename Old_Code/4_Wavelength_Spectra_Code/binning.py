import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 100, 100)
coarse_x = np.linspace(0, 100, 10)

mid_x = np.empty(len(coarse_x) - 1)


noise = np.random.randn(len(x))




coarse_noise = np.empty(len(coarse_x) - 1)





print(np.mean(noise))

for i in range(1, len(coarse_x)):

    
    ind = (x > coarse_x[i-1]) * (x <= coarse_x[i])
    
    coarse_noise[i-1] = np.mean(noise[ind])
    
    mid_x[i-1] = np.mean([coarse_x[i-1], coarse_x[i]])
    
    
plt.plot(x, noise)
plt.plot(mid_x, coarse_noise)

for i in coarse_x:
    plt.axvline(i, 0, 1, color='black', linestyle='--')

plt.show()