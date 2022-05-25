from PIL import Image
import pytesseract
import os
import cv2

image = cv2.imread('1.jpeg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
preprocess ='blur'
if preprocess =='thresh':
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
if preprocess =='blur':
    gray = cv2.medianBlur(gray, 3)

filename = '{}.png'.format(os.getpid())
cv2.imwrite(filename, gray)

text = pytesseract.image_to_string(Image.open(filename))
print(text)
os.remove(filename)

cv2.imshow('image', image)
cv2.imshow('gray', gray)
cv2.waitKey(0)

