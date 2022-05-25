import cv2
import numpy as np

im_source = cv2.imread('cocacola.png')
im_search = cv2.imread('bottle.png')
# 将图像和模板都转换为灰度
im_search_gray = cv2.cvtColor(im_search, cv2.COLOR_BGR2GRAY)
im_source_gray = cv2.cvtColor(im_source, cv2.COLOR_BGR2GRAY)
# sift = cv2.SIFT(threshold=0.8, rgb=True, nfeatures=50000)
sift = cv2.SIFT.create()
kp_sch, des_sch = sift.detectAndCompute(im_search, None)
kp_src, des_src = sift.detectAndCompute(im_source, None)

kplImgA = cv2.drawKeypoints(im_search_gray, kp_sch, im_search)
kplImgB = cv2.drawKeypoints(im_source_gray, kp_src, im_source)

# cv2.imshow("kplImgA", kplImgA)
# cv2.imshow("kplImgB", kplImgB)

# cv2.waitKey(0)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
# 匹配两个图片中的特征点集，k=2表示每个特征点取出2个最匹配的对应点:
# SIFT识别特征点匹配，参数设置:
FLANN_INDEX_KDTREE = 0
FLANN = cv2.FlannBasedMatcher({'algorithm': FLANN_INDEX_KDTREE, 'trees': 5}, dict(checks=50))# 匹配两个图片中的特征点集，k=2表示每个特征点取出2个最匹配的对应点:
matches = FLANN.knnMatch(des_sch, des_src, k=2)
# print(matches)
good_ratio = 0.59
good = []
# good为特征点初选结果，剔除掉前两名匹配太接近的特征点，不是独特优秀的特征点直接筛除(多目标识别情况直接不适用)
for m, n in matches:
    # print(m.distance)
    # print(n.distance)
    if m.distance < good_ratio * n.distance:
        good.append(m)
# print(good)
# good点需要去除重复的部分，（设定源图像不能有重复点）去重时将src图像中的重复点找出即可
# 去重策略：允许搜索图像对源图像的特征点映射一对多，不允许多对一重复（即不能源图像上一个点对应搜索图像的多个点）
good_diff, diff_good_point = [], [[]]
for m in good:
    diff_point = [int(kp_src[m.trainIdx].pt[0]), int(kp_src[m.trainIdx].pt[1])]
    if diff_point not in diff_good_point:
        good_diff.append(m)
        diff_good_point.append(diff_point)
good = good_diff
# print(good)
# for i in good:
#     print(i.distance)

def _find_homography(sch_pts, src_pts):
    """多组特征点对时，求取单向性矩阵."""
    try:
        M, mask = cv2.findHomography(sch_pts, src_pts, cv2.RANSAC, 5.0)
    except Exception:
        import traceback
        traceback.print_exc()
        raise Exception("OpenCV error in _find_homography()...")
    else:
        if mask is None:
            raise Exception("In _find_homography(), find no mask...")
        else:
            return M, mask

def _many_good_pts(im_source, im_search, kp_sch, kp_src, good):
    """特征点匹配点对数目>=4个，可使用单矩阵映射,求出识别的目标区域."""
    sch_pts, img_pts = np.float32([kp_sch[m.queryIdx].pt for m in good]).reshape(
        -1, 1, 2), np.float32([kp_src[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    # M是转化矩阵
    M, mask = _find_homography(sch_pts, img_pts)
    matches_mask = mask.ravel().tolist()
    # 从good中间筛选出更精确的点(假设good中大部分点为正确的，由ratio=0.7保障)
    selected = [v for k, v in enumerate(good) if matches_mask[k]]

    # 针对所有的selected点再次计算出更精确的转化矩阵M来
    sch_pts, img_pts = np.float32([kp_sch[m.queryIdx].pt for m in selected]).reshape(
        -1, 1, 2), np.float32([kp_src[m.trainIdx].pt for m in selected]).reshape(-1, 1, 2)
    M, mask = _find_homography(sch_pts, img_pts)
    # print(M, mask)
    # 计算四个角矩阵变换后的坐标，也就是在大图中的目标区域的顶点坐标:
    h, w = im_search.shape[:2]
    h_s, w_s = im_source.shape[:2]
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    # trans numpy arrary to python list: [(a, b), (a1, b1), ...]
    def cal_rect_pts(dst):
        return [tuple(npt[0]) for npt in dst.astype(int).tolist()]

    pypts = cal_rect_pts(dst)
    # 注意：虽然4个角点有可能越出source图边界，但是(根据精确化映射单映射矩阵M线性机制)中点不会越出边界
    lt, br = pypts[0], pypts[2]
    middle_point = int((lt[0] + br[0]) / 2), int((lt[1] + br[1]) / 2)
    # 考虑到算出的目标矩阵有可能是翻转的情况，必须进行一次处理，确保映射后的“左上角”在图片中也是左上角点：
    x_min, x_max = min(lt[0], br[0]), max(lt[0], br[0])
    y_min, y_max = min(lt[1], br[1]), max(lt[1], br[1])
    # 挑选出目标矩形区域可能会有越界情况，越界时直接将其置为边界：
    # 超出左边界取0，超出右边界取w_s-1，超出下边界取0，超出上边界取h_s-1
    # 当x_min小于0时，取0。  x_max小于0时，取0。
    x_min, x_max = int(max(x_min, 0)), int(max(x_max, 0))
    # 当x_min大于w_s时，取值w_s-1。  x_max大于w_s-1时，取w_s-1。
    x_min, x_max = int(min(x_min, w_s - 1)), int(min(x_max, w_s - 1))
    # 当y_min小于0时，取0。  y_max小于0时，取0。
    y_min, y_max = int(max(y_min, 0)), int(max(y_max, 0))
    # 当y_min大于h_s时，取值h_s-1。  y_max大于h_s-1时，取h_s-1。
    y_min, y_max = int(min(y_min, h_s - 1)), int(min(y_max, h_s - 1))
    # 目标区域的角点，按左上、左下、右下、右上点序：(x_min,y_min)(x_min,y_max)(x_max,y_max)(x_max,y_min)
    pts = np.float32([[x_min, y_min], [x_min, y_max], [
                     x_max, y_max], [x_max, y_min]]).reshape(-1, 1, 2)
    pypts = cal_rect_pts(pts)

    return middle_point, pypts, [x_min, x_max, y_min, y_max, w, h]


# 匹配点对 >= 4个，使用单矩阵映射求出目标区域，据此算出可信度：
middle_point, pypts, w_h_range = _many_good_pts(im_source, im_search, kp_sch, kp_src, good)

print(middle_point)
print(pypts)
print(w_h_range)
# best_match = generate_result(middle_point, pypts, confidence)
#
# print("[sift] result=%s" % (best_match))


# matchesMask = [[0, 0] for i in range(len(matches))]
# coff = 0.2
# for i,(m,n) in enumerate(matches):
#     if m.distance < coff * n.distance:
#         matchesMask[i]=[1,0]
#
# print(matchesMask)
# draw_params = dict(matchColor = (0,255,0),
#                    singlePointColor = (255,0,0),
#                    matchesMask = matchesMask,
#                    flags = 0)
# resultImg = cv2.drawMatchesKnn(imageGray, keypointsA, templateGray,keypointsB, matches,None,**draw_params)
# # resultImg1 = cv2.drawMatchesKnn(image, keypointsA, template,keypointsB, matches,None,**draw_params)
# cv2.imshow("resultImg", resultImg)
# # cv2.imshow("resultImg1", resultImg1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()