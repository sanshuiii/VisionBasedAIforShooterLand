import numpy as np
import cv2
from utils.ground_seg import img2dis

# 自己头上的绿色血条    [55,230,200] - [65,240,210]
# 怪物头上的红色血条    [3,240,178] - [10,255,255] 小怪
# 障碍物1 [5,182,100] - [45,230,235]
# 障碍物2 [0,0,0] - [35,60,235]

imgsForIdentification = [
    img2dis(np.array(cv2.resize(cv2.imread('picture/win.png'),(270,585)))),
    img2dis(np.array(cv2.resize(cv2.imread('picture/upgrade.png'),(270,585)))),
    img2dis(np.array(cv2.resize(cv2.imread('picture/fairy.png'),(270,585)))),
    img2dis(np.array(cv2.resize(cv2.imread('picture/killed.png'),(270,585)))),
    img2dis(np.array(cv2.resize(cv2.imread('picture/end.png'),(270,585))))
]

print(imgsForIdentification)

'''通过大面积蓝色确定进入技能选择界面'''
def is_select_skill(img):
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    low = [100,170,0]
    high = [255,255,255]
    mask = cv2.inRange(hsv, np.array(low), np.array(high))
    target = len(mask[mask==255])
    bg = len(mask[mask==0])
    # 还要判断是不是最后一关前的女神祝福
    if target/(target+bg) < 0.15:
        return -1 # 默认状态
    img = np.array(img)
    img = img2dis(img)
    dis = []
    for i in range(len(imgsForIdentification)):
        dis.append(np.mean(np.power(img-imgsForIdentification[i],2)))
    dis = np.array(dis)
    return np.argmin(dis)