import cv2
template = cv2.imread('cocacola.png')
image = cv2.imread('bottle.png')
# 将图像和模板都转换为灰度
imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

result = cv2.matchTemplate(imageGray, templateGray,	cv2.TM_CCOEFF_NORMED)
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)
print(cv2.minMaxLoc(result))
# 确定起点和终点的（x，y）坐标边界框
(startX, startY) = maxLoc
endX = startX + template.shape[1]
endY = startY + template.shape[0]

# 在图像上绘制边框
cv2.rectangle(image, (startX, startY), (endX, endY), (255, 0, 0), 3)
# 显示输出图像
cv2.imshow("Output", image)
cv2.waitKey(0)


