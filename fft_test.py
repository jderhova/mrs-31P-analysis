import numpy as np

import matplotlib.pyplot as plt

 

# Time period

t = np.arange(0, 10, 0.01);

f1= np.sin(2*np.pi*2*t) + np.sin(2*np.pi*t)

ft= f1*np.exp(-1j*2*np.pi*3*t)

plt.plot(ft.real, ft.imag)
plt.show()