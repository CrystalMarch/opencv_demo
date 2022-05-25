# https://blog.csdn.net/weixin_56701689/article/details/118577276

import cv2
import numpy as np
def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)  # 等待时间，毫秒级
    cv2.destroyAllWindow()

image = cv2.imread('1.jpeg')
ratio = image.shape[0] / 500.0
#为什么是500.0 是因为我想他换H=500的衣服，看看比率
#让我们写个函数 教他怎么换衣服
def resize(image, width=None, height=None):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(h)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, cv2.INTER_AREA)
    return resized
#换上了品如的衣服：
orig=image.copy()  #后面还要copy
image = resize(orig, width=None, height=500)

# 预处理
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv_show('gray', gray)


gray = cv2.GaussianBlur(gray, (5, 5), 0)
#高斯滤波去掉噪声点
edged = cv2.Canny(gray, 75, 200)
cv_show('edged', edged)
#经典Canny检测  原形毕露！何方妖孽！！！

# 轮廓检测
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]

cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
#我这里key定义的是面积 我就取4个 因为轮廓检测是很多的，必然包括图上的小票轮廓，排个序，取前五个，之后我再设置一定的规则去筛选这几个轮廓

for i in cnts:

    # 计算轮廓近似  peri是周长  老盆友
    peri = cv2.arcLength(i, True)
    #   c表示输入的点集合
    #   epsilon 表示从原始轮廓到近似轮廓的最大距离，它是一个准确度参数
    #   True表示封闭的
    approx = cv2.approxPolyDP(i, 0.02 * peri, True)
    screenCnt = []
    # 4个点时候就拿出来  得到矩形
    if len(approx) == 4:
        screenCnt = approx
        break
print(screenCnt)

cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv_show('image', image)


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    # 0 0
    # 0 0
    # 0 0
    # 0 0

    s = pts.sum(axis=1)  # 计算矩阵的每一行元素相加之和
    rect[0] = pts[np.argmin(s)]  # tl
    rect[2] = pts[np.argmax(s)]  # bl

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # tr
    rect[3] = pts[np.argmax(diff)]  # br
    # return the ordered coordinates
    return rect


def four_point_tramsform(image, pts):
    # 获取输入的坐标
    rect = order_points(pts)
    (tl, tr, bl, br) = rect

    # 计算出输入的w和h的值  勾股定理  走起
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # 变换之后的坐标位置
    # 计算变换矩阵M  rect是轮廓四个点  dst是我们规定的四个点（利用W和H人为创造的）
    M = cv2.getPerspectiveTransform(rect, dst)

    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))


    # 返回变换后的结果  杰哥：让我康康！！
    return warped

warped = four_point_tramsform(orig, screenCnt.reshape(4, 2) * ratio)
# screenCnt是 Canny检测得到的  因为缩放 每个点的位置都要改
# 别忘了*比率 因为我们现在这个图orig是初始image的复制体
cv_show('warped', warped)

# 二值化处理
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
ref = cv2.threshold(warped, 100, 255, cv2.THRESH_BINARY)[1]
cv2.imwrite('scan.jpg', ref)

# 展示结果
cv2.imshow('Original', resize(orig, height=650))
cv2.imshow('Scanned', resize(ref, height=650))
cv2.waitKey(0)


