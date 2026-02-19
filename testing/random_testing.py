import numpy as np
import matplotlib.pyplot as plt

height = 100
width = 100

mat = np.random.randint(0, 255, height * width).reshape(height, width)
plt.imshow(mat, cmap='Greens')
plt.show()
