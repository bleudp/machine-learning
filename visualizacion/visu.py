import matplotlib.pyplot as plt
import numpy as np
t = np.arange(0.0,5.0,0.01)
s = np.cos(2*np.pi*t)
line, = plt.plot(t, s, lw=2)

plt.ylim(-2,2)
plt.show()
