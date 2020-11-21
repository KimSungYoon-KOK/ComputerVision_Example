import matplotlib.pyplot as plt
import numpy as np
import cv2

image = cv2.imread('/Users/kok_ksy/Documents/GitHub/CV_example/Exercise1/neutrophils.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

pixel_values = image.reshape((-1, 3))
pixel_values = np.float32(pixel_values)
print(pixel_values.shape)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER, 100, 0.2)

k=3
_, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
centers = np.uint8(centers)
labels = labels.flatten()

segmented_image = centers[labels.flatten()]

segmented_image = segmented_image.reshape(image.shape)

plt.figure('Kmeans Image')
plt.subplot(211)
plt.title('Original Image')
plt.imshow(image)
plt.subplot(212)
plt.title('segmented_image')
plt.imshow(segmented_image)
plt.show()