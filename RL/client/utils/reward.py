import numpy as np
# from cnocr import CnOcr
from paddleocr import PaddleOCR, draw_ocr
import cv2 as cv
import time
import os
import re

# low: 55 230 200
# high: 65 240 210

LowH = 55
HighH = 65
LowS = 230
HighS = 240
LowV = 200
HighV = 210

REWARD_UP_DOWN = 0.1
LOSS_BLOOD = -10
REWARD_PASS_GAME = 10

k = np.ones((5, 5), np.uint8)


class REWARD():
    def __init__(self):
        self.wait_for_num = 0

    def cal_reword1(self, img_old, img_new):
        # 1. 血条检测
        img_old_hsv = cv.cvtColor(img_old, cv.COLOR_BGR2HSV)
        img_new_hsv = cv.cvtColor(img_new, cv.COLOR_BGR2HSV)

        img_old_target = cv.inRange(img_old_hsv, (LowH, LowS, LowV), (HighH, HighS, HighV))
        img_new_target = cv.inRange(img_new_hsv, (LowH, LowS, LowV), (HighH, HighS, HighV))

        # 闭运算
        morph_open_old = cv.morphologyEx(img_old_target, cv.MORPH_CLOSE, kernel=k)
        morph_open_new = cv.morphologyEx(img_new_target, cv.MORPH_CLOSE, kernel=k)

        # cv.imwrite("morph_old.png", morph_open_old)
        # cv.imwrite("morph_new.png", morph_open_new)
        # 从二值图像中提取轮廓
        # contours中包含检测到的所有轮廓,以及每个轮廓的坐标点
        contours_old = cv.findContours(morph_open_old.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[1]
        contours_new = cv.findContours(morph_open_new.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[1]

        return contours_old, contours_new

    def cal_reword(self, ocr, img_old, img_new):
        flag = 0
        # print("in reward function")
        contours_old, contours_new = self.cal_reword1(img_old, img_new)
        # 遍历检测到的所有轮廓,并将检测到的坐标点画在图像上
        # c的类型numpy.ndarray,维度(num, 1, 2), num表示有多少个坐标点
        '''
        for c in contours_old:
            cv.drawContours(img_old, [c], -1, (0, 0, 255), 1)
        '''
        try:
            x_old, y_old, w_old, h_old = cv.boundingRect(contours_old[0])
            x_new, y_new, w_new, h_new = cv.boundingRect(contours_new[0])
            roi_old = img_old[y_old - h_old - 10: y_old, x_old: x_old + w_old]
            roi_new = img_new[y_new - h_new - 10: y_new, x_new: x_new + w_new]

            roi_old = cv.cvtColor(roi_old, cv.COLOR_BGR2GRAY)
            result_old = ocr.ocr(roi_old, cls=True)
            roi_new = cv.cvtColor(roi_new, cv.COLOR_BGR2GRAY)
            result_new = ocr.ocr(roi_new, cls=True)
            blood_old_str = re.findall(r"\d+\.?\d*", result_old[0][-1][0])
            blood_old = int(''.join(str(i) for i in blood_old_str))
            print(blood_old)
            blood_new_str = re.findall(r"\d+\.?\d*", result_new[0][-1][0])
            blood_new = int(''.join(str(i) for i in blood_new_str))
            print(blood_new)
            if  blood_old - 5000 < blood_new < blood_old - 1000:  # 扣血在一定范围内，防止OCR识别的误差
                reward1 = LOSS_BLOOD
            else:  # 其他情况默认为不扣血
                reward1 = 0

        except:
            ct = time.time()
            data_head = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(ct))
            data_secs = (ct - int(ct)) * 1000
            time_stamp = "%s-%03d" % (data_head, data_secs)
            fname = time_stamp + '.png'
            # cv.imwrite(os.path.join('./badcase/old-', fname), img_old)
            # cv.imwrite(os.path.join('./badcase/new-', fname), img_new)
            return 0

        '''
        # w 为血条宽度
        if w_old == 0 or w_new == 0:
            reward1 = 0
        else:
            error_blood = w_new - w_old
            if error_blood < 0:  # 扣血
                reward1 = -10
            elif error_blood >= 0:  # 不扣血
                reward1 = 0
        '''

        # 2. 光流检测
        # ShiTomasi 角点检测参数
        # maxCorners : 设置最多返回的关键点数量。
        # qualityLevel : 反应一个像素点强度有多强才能成为关键点。
        # minDistance : 关键点之间的最少像素点。
        # blockSize : 计算一个像素点是否为关键点时所取的区域大小。
        # useHarrisDetector :使用原声的 Harris 角侦测器或最小特征值标准。
        begin_time = time.time()
        feature_params = dict(maxCorners=100,
                              qualityLevel=0.1,
                              minDistance=7,
                              blockSize=7)

        #  lucas kanade光流法参数
        lk_params = dict(winSize=(50, 50),  # 搜索窗口的大小
                         maxLevel=2,
                         criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.01))

        # 创建随机光流颜色
        color = np.random.randint(0, 255, (100, 3))

        # 找到原始灰度图
        # img_old = c
        gray_old = cv.cvtColor(img_old, cv.COLOR_BGR2GRAY)

        # 获取图像中的角点，返回到p0中
        p0 = cv.goodFeaturesToTrack(gray_old, mask=None, **feature_params)

        # 创建一个蒙版用来画轨迹
        mask = np.zeros_like(img_old)

        # img_new = cv.imread("test_img/test_5.png")
        gray_new = cv.cvtColor(img_new, cv.COLOR_BGR2GRAY)

        # 计算光流
        p1, st, err = cv.calcOpticalFlowPyrLK(gray_old, gray_new, p0, None, **lk_params)

        # 选取好的跟踪点
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        end_time = time.time()
        time_cost = end_time - begin_time
        # print("time cost: " + str(time_cost))

        error = good_new - good_old

        # 第二列是y轴
        # print("error x-axis mean: " + str(error[:, 0].mean()))
        if error[:, 1].mean() > 0:
            # 背景向下移动，说明人向上走
            if -5 < error[:, 0].mean() < 5:
                reword2 = REWARD_UP_DOWN
            else:
                reword2 = 0
        elif error[:, 1].mean() <= 0:
            # 背景向上移动，说明人向下走
            reword2 = -REWARD_UP_DOWN

        # 3. 判断关卡，如果通关给10个reward
        # print(gray_new.shape)
        roi_new = gray_new[int(30 / 0.375):int(42 / 0.375), int(115 / 0.375):int(155 / 0.375)]
        ret, _ = cv.threshold(roi_new, 230, 255, cv.THRESH_BINARY)
        res_new = ocr.ocr(_, cls=True)
        res_new = res_new[0][-1][0]
        roi_old = gray_old[int(30 / 0.375):int(42 / 0.375), int(115 / 0.375):int(155 / 0.375)]
        ret, _ = cv.threshold(roi_old, 230, 255, cv.THRESH_BINARY)
        res_old = ocr.ocr(_, cls=True)
        res_old = res_old[0][-1][0]
        # global i
        # i = i + 1
        # cv.imwrite("data/"+str(i)+".png", np.array(gray_new))
        if len(res_new) > 2:
            if self.wait_for_num == 5 and ("最" in res_new or "终" in res_new):
                print("通关！！！")
                reward3 = REWARD_PASS_GAME
                self.wait_for_num += 1
                flag = 1
            if "关卡" + str(self.wait_for_num) + "/5" == res_new:
                print("通关！！！")
                reward3 = REWARD_PASS_GAME
                self.wait_for_num += 1
                flag = 1

            else:
                reward3 = 0
        else:
            reward3 = 0
        total_reward = reward1 + reword2 + reward3
        # print("total_reward: " +str(total_reward) + " reward2: " + str(reword2) + " reward3: " + str(reward3))

        return total_reward, flag

        # print(w_new)
        # print(w_old)
        # print("==========")
        # if reward1 != 0:
        #     print("--------------------")
        #     print("hp: "+str(reward1))
        #     print("mv: "+str(reword2))

        # return total_reward

        # 画出轨迹
        '''
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()  # 将数组维度拉成一维
            c, d = old.ravel()
            a = int(a)
            b = int(b)
            c = int(c)
            d = int(d)
            mask = cv.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            frame = cv.circle(img_new, (a, b), 5, color[i].tolist(), -1)
        img = cv.add(frame, mask)
        cv.imshow('test', img)
        cv.waitKey(0)
        '''


if __name__ == '__main__':
    img_old_ = cv.imread("test_8_3/11.png")
    img_new_ = cv.imread("test_8_3/12.png")
    # img_old = cv.resize(img_old, (128, 128))
    # img_new = cv.resize(img_new, (128, 128))

    print(img_old_.shape)
    # 传入1560 * 720的图
    # myocr = CnOcr()
    myReward = REWARD()
    myocr = PaddleOCR(use_angle_cls=True, lang='ch', use_gpu=False)  # need to run only once to download and load model into memory
    myreward = myReward.cal_reword(myocr, img_old_, img_new_)
    print(myreward)
