import imageio
import numpy as np
import matplotlib.pyplot as plt
img = imageio.imread("/ssd1/jiyu/data/unity/rs55/set7/000000000-outlineSegmentation.png")
print(img.shape, img.max(), img.min())
print(np.unique(img))

plt.imshow(img)
plt.axis("off")
plt.savefig("image.png")
plt.show()