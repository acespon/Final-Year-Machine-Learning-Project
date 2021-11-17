import numpy
import cv2
import os
import pandas
import matplotlib


print('hello world!')



for files in os.listdir("./images/images/Andy_Warhol"):
  print(files)
  filesD = cv2.imread("./images/images/Andy_Warhol/" + str(files))

  cv2.imshow('image', filesD)
  cv2.waitKey(0)


# filesD = cv2.imread("./images/images/Andy_Warhol/Andy_Warhol_1.jpg")
# cv2.imshow('image', filesD)
# cv2.waitKey(0)



