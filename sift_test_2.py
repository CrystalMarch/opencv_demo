import numpy as np
import cv2

MIN_MATCH_COUNT = 10

img1 = cv2.imread('cocacola.png')          # queryImage
img2 = cv2.imread('bottle.png')     # trainImage
sift = cv2.SIFT.create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    # 获取两组点间的转化
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    h, w = img1.shape[:2]
    print(w, h)
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    # 计算得到转换矩阵
    dst = cv2.perspectiveTransform(pts, M,)
    # 使用warpPerspective()进行透视变换
    # img_w = cv2.warpPerspective(img2, M, (w2+w, h2))
    # cv2.imshow("img_w", img_w)
    img_pig = cv2.imread('bottle_pig.png')
    img4 = cv2.warpPerspective(img_pig, np.linalg.inv(M), (w, h))
    img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    cv2.imshow("img_w", img4)

else:
    print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
        singlePointColor = None,
        matchesMask = matchesMask, # draw only inliers
        flags = 2)

img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

cv2.imshow("gray", img3)
cv2.waitKey(0)
