# import cv2
# print(cv2.getBuildInformation())

import cv2
print(cv2.__version__)
img = cv2.imread('path_to_an_image.jpg')
cv2.imshow('Test Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
