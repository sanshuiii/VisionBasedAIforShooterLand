import os,sys,subprocess,threading, time, signal
import argparse
import random
from collections import deque
import numpy as np
import cv2
from brain import *
from cnocr import CnOcr
import random
from PIL import Image
myocr = CnOcr()


def quit(signum, frame):
    print('Ctrl-C退出所有线程')
    sys.exit()
K = 1.5
action_list = [
    f'input tap {550//K} {1450//K}', # 原地不动射击
    f'input swipe {550//K} {1850//K} {550//K} {1500//K} 500', # 前
    f'input swipe {550//K} {1850//K} {200//K} {1850//K} 500', # 左
    f'input swipe {550//K} {1850//K} {900//K} {1850//K} 500', # 右
    f'input swipe {550//K} {1850//K} {550//K} {2200//K} 500', # 后
    f'input tap {550//K} {1850//K}', # 放弃女神的祝福
]

action_queue = deque()
in_stage_4 = False
after_pool = False


def make_regalur_image(img, size=(256, 256)):
    return img.resize(size).convert('RGB')

def hist_similar(lh, rh):
    assert len(lh) == len(rh)
    return sum(1 - (0 if l == r else float(abs(l - r)) / max(l, r)) for l, r in zip(lh, rh)) / len(lh)


def calc_similar(li, ri):
    return hist_similar(li.histogram(), ri.histogram())


def getAction(img_pool):
    global action_queue
    global in_stage_4
    global after_pool
    while 1:
        # 动作队列未执行完，继续执行。
        if len(action_queue) != 0:
            continue
        # 动作队列为空，进行新一轮决策。
        if os.path.exists('tmp.png'):
            # print('截图已经接收')
            time.sleep(0.05)
            img_origin = cv2.imread('tmp.png')
            # img_gray = cv2.cvtColor(img_origin, cv2.COLOR_BGR2GRAY)
            try:
                roi = img_origin[int(30 / 0.375):int(42 / 0.375), int(115 / 0.375):int(155 / 0.375)]
                ret, _ = cv2.threshold(roi, 220, 255, cv2.THRESH_BINARY)
                res = myocr.ocr_for_single_line(_)
                # print(res)
                # 如果进入第四关， 切换控制器
                if "4" in res or "最" in res or "终" in res:
                    in_stage_4 = True
                else:
                    in_stage_4 = False
                img = cv2.resize(img_origin, (270, 585))
                print("####")
                # img = cv2.resize(img,None,fx=0.25,fy=0.25)
                # 看是否到达池塘
                pil_img_pool = Image.fromarray(cv2.cvtColor(img_pool, cv2.COLOR_BGR2RGB))
                pil_img_origin = Image.fromarray(cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB))
                similarity = calc_similar(pil_img_pool, pil_img_origin)
                print(similarity)
                if "5" in res and similarity > 0.70:
                    print("来到池塘, 转换控制器")
                    after_pool = True
            except:
                continue
            os.system('rm tmp.png')

            skillFlag, lastStageFlag = is_select_skill(img)
            
            if skillFlag and not lastStageFlag:
                print('升级，进入技能选择界面...')
                action_queue.extend([0, 0])
            elif skillFlag and lastStageFlag:
                print('女神的祝福')
                # action_queue.extend([5, 5, 5, 5])
                # action_queue.extend([1]*20)

            else:
                planner = Planner(max_length=8, img=img, w=25, h=25, debug=0, in_stage_4=in_stage_4, after_pool=after_pool)
                action_list = planner.get_result()
                print(action_list)
                action_queue.extend(action_list)
                if is_under_attack(img):
                    print('战斗状态，额外打两枪')
                    action_queue.extend([0, 0])
                # 以时间命名截图，毫秒级
                # ct = time.time()
                # data_head = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(ct))
                # data_secs = (ct - int(ct)) * 1000
                # time_stamp = "%s-%03d" % (data_head, data_secs)
                # fname = time_stamp+'.png'
                # cv2.imwrite(os.path.join('./2',fname),img)

            
parser = argparse.ArgumentParser(description='与射手大陆多线程交互')
parser.add_argument('--autoStart', type=bool, default=1, help='是否自动开局')
args = parser.parse_args()

# record = subprocess.Popen(['adb','shell','screenrecord --time-limit 150 /sdcard/demo.mp4'])
if __name__ == '__main__':

    signal.signal(signal.SIGINT, quit)
    signal.signal(signal.SIGTERM, quit)
    
    if args.autoStart:
        ret = os.popen('adb shell dumpsys activity activities | grep com.lightboat.bowland').read()
        if not ret:
            print('未检测到射手大陆启动，游戏自启动中...')
            os.system('adb root')
            os.system('adb shell am start -n com.lightboat.bowland.android.ohayoo/.CustomUnityPlayerActivity')
            print('游戏启动后，手动把各种礼包领完，确认进入初始页面后重新运行脚本')
            sys.exit()
        else:
            # 自动开局，写死
            print('游戏加载中...')
            os.system(f'adb shell input tap {750//K} {420//K}') # 1-3冻霜关卡图标
            os.system(f'adb shell input tap {750//K} {1950//K}') # “开始”图标
            time.sleep(3) # 等待动画效果
            # 开局自动选技能，写死
            print('技能选择中...')
            os.system(f'adb shell input swipe {550//K} {850//K} {550//K} {500//K} 1000') # 直走1秒自动打开选择界面
            time.sleep(2) # 等待动画效果
            os.system(f'adb shell input tap {700//K} {1900//K}') # 第一个技能图标图标380,第二个700
            os.system(f'adb shell input tap {700//K} {1900//K}') # 一定要tap两次，第一次选中，第二次确认
            print('准备开始游戏...')
            time.sleep(1)

    from utils import imgcallback # 图像回传线程
    t1 = threading.Thread(target=imgcallback.getImg)
    t1.setDaemon(True)
    t1.start()

    img_pool = cv2.imread("pool.png")
    t2 = threading.Thread(target=getAction, args=(img_pool, ))
    t2.setDaemon(True)
    t2.start()

    # 策略执行代码 
    while True:
        if len(action_queue):
            idx = action_queue[0]
            sp = subprocess.Popen(['adb', 'shell', action_list[idx]])
            time.sleep(0.45)
            action_queue.popleft()
            sp.terminate()
        else:
            sp = subprocess.Popen(['adb', 'shell', action_list[0]])
            time.sleep(0.45)
            sp.terminate()
        '''
        random_num = random.randint(0, 20)
        if random_num == 1:
            # 向后，左走
            print("向左走")
            sp = subprocess.Popen(['adb', 'shell', action_list[2]])
            time.sleep(0.45)
            sp.terminate()
        elif  random_num == 2:
            # 向右走
            print("向右走")
            sp = subprocess.Popen(['adb', 'shell', action_list[3]])
            time.sleep(0.45)
            sp.terminate()
        elif random_num == 3:
            # 向右走
            print("向后走")
            sp = subprocess.Popen(['adb', 'shell', action_list[4]])
            time.sleep(0.45)
            sp.terminate()
        else :
            if len(action_queue):
                idx = action_queue[0]
                sp = subprocess.Popen(['adb','shell',action_list[idx]])
                time.sleep(0.45)
                action_queue.popleft()
                sp.terminate()
            else:
                sp = subprocess.Popen(['adb','shell',action_list[0]])
                time.sleep(0.45)
                sp.terminate()
        '''
