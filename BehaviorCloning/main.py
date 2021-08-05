import os,sys,subprocess,threading, time, signal
import argparse
from collections import deque
import numpy as np
import cv2
from utils.brain import *
from copy import deepcopy


import torch
from torchvision.models import mobilenet_v2
from torchvision import transforms
transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
test_model = mobilenet_v2(pretrained=False,num_classes=5).to(device)
test_model.load_state_dict(torch.load('./model/1_1_BC_v1.pth'))
test_model.eval()


def quit(signum, frame):
    print('Ctrl-C退出所有线程')
    sys.exit()

action_list = [
    'input tap 550 1450', # 原地不动射击
    'input swipe 550 1850 550 1500 500', # 前
    'input swipe 550 1850 200 1850 500', # 左
    'input swipe 550 1850 900 1850 500', # 右
    'input swipe 550 1850 550 2200 500', # 后
    'input tap 550 1850', # 放弃女神的祝福
]

action = 1

def getAction():
    global action
    while 1:
        if os.path.exists('tmp.png'):
            #print('截图已经接收')
            time.sleep(0.05)
            img = cv2.imread('tmp.png')
            try:
                img = cv2.resize(img,None,fx=0.25,fy=0.25)
            except:
                continue
            os.system('rm -r tmp.png')
            skillFlag, lastStageFlag = is_select_skill(img)
            if skillFlag and not lastStageFlag:
                print('升级，进入技能选择界面...')
                action = 0
            elif skillFlag and lastStageFlag:
                print('女神的祝福')
                action = 5
            else:
                img_ = deepcopy(img)
                img = cv2.resize(img,(128,128))
                img = np.asarray(img) / 255.0

                img = transform(img).float().to(device)
                img = torch.unsqueeze(img, 0)
                output = test_model(img)
                #print(output)
                _, action = torch.max(output.data, 1)
                action = action.item()
                # 以时间命名截图，毫秒级
                # ct = time.time()
                # data_head = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(ct))
                # data_secs = (ct - int(ct)) * 1000
                # time_stamp = "%s-%03d" % (data_head, data_secs)
                # fname = time_stamp+'_'+str(action)+'.png'
                # cv2.imwrite(os.path.join('./demonstration',fname),img_)

            
parser = argparse.ArgumentParser(description='与射手大陆多线程交互')
parser.add_argument('--autoStart', type=bool, default=1, help='是否自动开局')
args = parser.parse_args()
record = subprocess.Popen(['adb','shell','screenrecord --time-limit 130 /sdcard/demo.mp4'])
if __name__ == '__main__':

    signal.signal(signal.SIGINT, quit)
    signal.signal(signal.SIGTERM, quit)
    
    os.makedirs('demonstration', exist_ok=True)

    if args.autoStart:
        ret = os.popen('adb shell dumpsys activity activities | grep com.lightboat.bowland.mi').read()
        if not ret:
            print('未检测到射手大陆启动，游戏自启动中...')
            os.system('adb root')
            os.system('adb shell am start -n com.lightboat.bowland.mi/com.ohayoo.unifysdk.UnifySdkActivity')
            print('游戏启动后，手动把各种礼包领完，确认进入初始页面后重新运行脚本')
            sys.exit()
        else:
            # 自动开局，写死
            print('游戏加载中...')
            os.system('adb shell input tap 300 430') # 1-3冻霜关卡图标
            os.system('adb shell input tap 750 1950') # “开始”图标
            time.sleep(5) # 等待动画效果
            # 开局自动选技能，写死
            print('技能选择中...')
            os.system('adb shell input swipe 550 850 550 500 1000') # 直走1秒自动打开选择界面
            time.sleep(2) # 等待动画效果
            os.system('adb shell input tap 700 1900') # 第一个技能图标图标380,第二个700
            os.system('adb shell input tap 700 1900') # 一定要tap两次，第一次选中，第二次确认
            print('准备开始游戏...')
            time.sleep(1)

    from utils import imgcallback # 图像回传线程
    t1 = threading.Thread(target=imgcallback.getImg)
    t1.setDaemon(True)
    t1.start()

    t2 = threading.Thread(target=getAction)
    t2.setDaemon(True)
    t2.start()

    # from utils import joystick # 手柄线程
    # t3 = threading.Thread(target=joystick.getStick)
    # t3.setDaemon(True)
    # t3.start()

    # 策略执行代码 
    while True:
        sp = subprocess.Popen(['adb','shell',action_list[action]])
        time.sleep(0.45)
        sp.terminate()