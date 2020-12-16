import numpy as np
import cv2

img1 = cv2.imread('/Users/kok_ksy/Documents/GitHub/CV_example/Exercise1/box.png')
img2 = cv2.imread('/Users/kok_ksy/Documents/GitHub/CV_example/Exercise1/box_in_scene.png')

sift = cv2.xfeatures2d.SIFT_creat()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

good = []
for m,n in matches:
    if m.distance < 0.3*n.distance:
        good.append([m])

img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
cv2.imshow('image', img3)
cv2.waitKey()
cv2.destroyAllWindows()