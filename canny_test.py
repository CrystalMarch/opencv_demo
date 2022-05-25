import cv2
import cv2 as cv
import math
# https://blog.csdn.net/weixin_51567891/article/details/123225299


def get_percentage(canny):
    sp = canny.shape
    print(sp)
    sz1 = sp[0]  # height(rows) of image
    sz2 = sp[1]  # width(colums) of image
    counter = 0
    for i in range(sz1):
        for j in range(sz2):
            rgb = canny[i][j]
            if rgb == 255:
                counter = counter + 1
    percentage = math.ceil(counter * 100 / (sz1 * sz2)) / 100
    print(percentage)
    return percentage


img = cv.imread('2.png')
# b = img[0][0][0]
# g = img[0][0][1]
# r = img[0][0][2]
# print(r, g, b)
# cv.imshow('Image',img)
# cv.waitKey(3000)
canny = cv.Canny(img, 125, 175)
# cv.imshow('Canny', canny)
# cv.waitKey(3000)

canny2 = cv.Canny(img, 290, 250)
# cv.imshow('Canny2', canny2)


get_percentage(canny)
get_percentage(canny2)

# 进行高斯模糊
# blur = cv.GaussianBlur(img, (7, 7), cv.BORDER_DEFAULT)
# cv.imshow('Blur',blur)
# cv.waitKey(3000)

# canny1 = cv.Canny(blur, 125, 175)
# cv.imshow('Canny Edges After Blur', canny1)
# cv.waitKey(3000)
# cv.destroyWindow()