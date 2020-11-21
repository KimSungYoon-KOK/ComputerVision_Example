import numpy as np
import cv2

image = cv2.imread('/Users/kok_ksy/Documents/GitHub/CV_example/Exercise1/building.jpg', cv2.IMREAD_GRAYSCALE)
harris = cv2.cornerHarris(image, blockSize=3, ksize=3, k=0.04)
harris_norm = cv2.normalize(harris, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
image2 = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

for y in range(1, harris_norm.shape[0]-1):
    for x in range(1, harris_norm.shape[1]-1):
        if harris_norm[y, x] > 120:
            if (harris[y, x] > harris[y-1, x] and harris[y, x] > harris[y+1, x] and harris[y, x] > harris[y, x-1] and harris[y, x] > harris[y, x+1]):
                cv2.circle(image2, (x,y), radius=5, color=(0,0,255), thickness=2)

cv2.imshow('image', image)
cv2.imshow('harris_norm', harris_norm)
cv2.imshow('output', image2)
cv2.waitKey()
cv2.destroyAllWindows()