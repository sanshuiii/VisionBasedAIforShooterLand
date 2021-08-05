import os,sys,subprocess,threading, time, signal, cv2
from utils.brain import *
from utils.reward import REWARD
import numpy as np
import pickle
from model.custom_model import ACCNNModel
from cnocr import CnOcr

'''
hyper parameters
'''
max_step = 100
target = 3

prefix_model = []

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

IN_GAME_SIGNAL = False
WAIT_FOR_INIT_SIGNAL = False
WIN_SIGNAL = False
EVAL_MODEL = False


K = 1.5

def quit(signum, frame):
    print('Ctrl-C退出所有线程')
    sys.exit()

action_list = [
    f'input tap {550//K} {1450//K}', # 原地不动射击
    f'input swipe {550//K} {1850//K} {550//K} {1500//K} 500', # 前
    f'input swipe {550//K} {1850//K} {200//K} {1850//K} 500', # 左
    f'input swipe {550//K} {1850//K} {900//K} {1850//K} 500', # 右
    f'input swipe {550//K} {1850//K} {550//K} {2200//K} 500', # 后
    f'input tap {550//K} {1850//K}', # 放弃女神的祝福
]

episode = []
action = 1

def getAction():
    global action
    global episode
    global IN_GAME_SIGNAL
    global WAIT_FOR_INIT_SIGNAL
    global WIN_SIGNAL
    global max_step
    global EVAL_MODEL

    whole_steps = []
    myocr = CnOcr()
    step_num = 0
    re = REWARD()
    re.wait_for_num = 1
    lst_img = None
    WIN_SIGNAL = False
    cur_step = 0

    model = ACCNNModel([128,128,3], 5)
    with open("model/0.pt","rb") as f:
        weights = pickle.load(f)
    model.set_weights(weights)

    while 1:
        # print("s")
        if IN_GAME_SIGNAL == False:
            # print("w")
            if step_num == 0:
                continue
            else:
                with open("data/episode_data.pkl","wb") as f:
                    pickle.dump(episode, f)
                whole_steps.append(step_num)
                np.savetxt("res.txt", np.array(whole_steps))
                step_num = 0
                episode = []
                re.wait_for_num = 1
                cur_step = 0
                WIN_SIGNAL = False
                lst_img = None

                #  model = ACCNNModel([128,128,3], 5)
                with open("model/0.pt","rb") as f:
                    weights = pickle.load(f)
                model.set_weights(weights)
                
            continue
        if os.path.exists('tmp.png'):
            time.sleep(0.05)
            try:
                img = cv2.imread('tmp.png')
                os.system('rm -r tmp.png')
            except:
                continue
            actionType = is_select_skill(cv2.resize(img,(270,585)))
            step_num = step_num + 1
            if actionType == 1:
                print('升级，进入技能选择界面...')
                action = 0
            elif actionType == 2:
                print('女神的祝福')
                action = 5
            else:

                r = 0

                if actionType == 3:
                    print('复活了')
                    time.sleep(2)
                    os.system(f'adb shell input tap {350} {900}') # 复活
                    os.system(f'adb shell input tap {350} {900}') # 取消暂停
                elif actionType == 4: # dead
                    print("死透了")
                    IN_GAME_SIGNAL = False
                    WAIT_FOR_INIT_SIGNAL == False
                    r = -100
                    time.sleep(5)
                    os.system(f'adb shell input tap {350} {1300}') # 继续游戏
                    os.system(f'adb shell input tap {350} {1200}') # 可能没有奖励，补点一下继续
                    time.sleep(10)
                    os.system(f'adb shell input tap {640} {390}') # x掉我要变强
                    time.sleep(0.5) # 结束后回到初始界面
                elif actionType == 0: # win
                    IN_GAME_SIGNAL = False
                    WIN_SIGNAL = True
                    WAIT_FOR_INIT_SIGNAL == False
                    print("胜利了")
                    r = 100
                    time.sleep(5)
                    os.system(f'adb shell input tap {350} {1300}') # 退出游戏
                    time.sleep(8)
                elif step_num > max_step: # timeout
                    IN_GAME_SIGNAL = False
                    WAIT_FOR_INIT_SIGNAL == False
                    print("超时了")
                    r = -100
                    time.sleep(1.5)
                    os.system(f'adb shell input tap {45} {100}') # 暂停游戏
                    time.sleep(0.5)
                    os.system(f'adb shell input tap {300} {1100}') # 退出游戏
                    time.sleep(0.5)
                    os.system(f'adb shell input tap {450} {880}') # 确认退出游戏
                    time.sleep(3)

                if lst_img is not None:
                    img2 = cv2.resize(img,(128,128))/255.0
                    img3 = np.resize(img2,(1,128,128,3))

                    if re.wait_for_num == target-1:
                        if EVAL_MODEL:
                            action_list, p_list, v_list, logit = model.forward2(img3)
                        else:
                            action_list, p_list, v_list, logit = model.forward(img3)
                    else:
                        action_list, p_list, v_list, logit = model.forward2(img3)


                    print(logit)
                    action = action_list[0]
                    p = p_list[0]
                    v = v_list[0]
                    dr, flg = re.cal_reword(myocr, img, lst_img)
                    r = r + dr -1
                    if action == 1:
                        r = r + 2

                    if re.wait_for_num == target:
                        IN_GAME_SIGNAL = False
                        WAIT_FOR_INIT_SIGNAL == False
                        r = r + 100
                        print("通关！")
                        time.sleep(1.5)
                        os.system(f'adb shell input tap {45} {100}') # 暂停游戏
                        time.sleep(0.5)
                        os.system(f'adb shell input tap {300} {1100}') # 退出游戏
                        time.sleep(0.5)
                        os.system(f'adb shell input tap {450} {880}') # 确认退出游戏
                        time.sleep(3)
                        
                    
                    tmp  = [img2, action,r,p,v,(True if IN_GAME_SIGNAL==False else False)]
                    if re.wait_for_num >= target-1:
                        episode.append(tmp)
                        WAIT_FOR_INIT_SIGNAL = True
                    
                    if re.wait_for_num != target and flg:
                        cur_step += 1
                        # model = ACCNNModel([128,128,3], 5)
                        with open("model/%d.pt"%cur_step,"rb") as f:
                            weights = pickle.load(f)
                        # print(weights)
                        print("=========stage %d======="%(cur_step))
                        model.set_weights(weights)
                    
                lst_img = img

def initGame():
    global WAIT_FOR_INIT_SIGNAL
    global IN_GAME_SIGNAL
    # 自动开局，写死
    print('游戏加载中...')
    os.system(f'adb shell input tap {350} {1450}') # 冒险
    time.sleep(0.5)
    os.system(f'adb shell input tap {600} {30}') # 购买体力
    time.sleep(0.5)
    os.system(f'adb shell input tap {250} {900}') # 体力+1
    time.sleep(0.5)
    os.system(f'adb shell input tap {360} {960}') # 确定
    time.sleep(0.5)
    os.system(f'adb shell input tap {200} {200}') # 1-1关卡图标
    time.sleep(0.5)
    os.system(f'adb shell input tap {750//K} {1950//K}') # “开始”图标
    time.sleep(5) # 等待动画效果
    # 开局自动选技能，写死
    print('技能选择中...')
    os.system(f'adb shell input swipe {550//K} {850//K} {550//K} {500//K} 1000') # 直走1秒自动打开选择界面
    time.sleep(2) # 等待动画效果
    os.system(f'adb shell input tap {700//K} {1900//K}') # 第一个技能图标图标380,第二个700
    os.system(f'adb shell input tap {700//K} {1900//K}') # 一定要tap两次，第一次选中，第二次确认
    print('准备开始游戏...')
    IN_GAME_SIGNAL = True
    time.sleep(1)

if __name__ == '__main__':

    signal.signal(signal.SIGINT, quit)
    signal.signal(signal.SIGTERM, quit)

    from utils import imgcallback # 图像回传线程
    t1 = threading.Thread(target=imgcallback.getImg)
    t1.setDaemon(True)
    t1.start()

    t2 = threading.Thread(target=getAction)
    t2.setDaemon(True)
    t2.start()

    
    epoch_num = 0 

    # for i in range(target - 2):
    #     prefix_model.append(ACCNNModel([128,128,3], 5))
    #     prefix_model[-1].set_weights("model/%d.pt"%i)



    while True:
        
        epoch_num = epoch_num + 1
        print("---------round%d----------"%epoch_num)

        os.system('echo ')

        os.system('scp root@123.57.173.37:/root/model/model.pt model/%d.pt'%(target-2))
        # with open("model/model.pt","rb") as f:
        #     weights = pickle.load(f)
        # model.set_weights(weights)
        # print("模型加载成功！")

        initGame() 
        while IN_GAME_SIGNAL:
            sp = subprocess.Popen(['adb','shell',action_list[action]])
            time.sleep(0.45)
            sp.terminate()
            # print("eeee")
        time.sleep(3)

        while WAIT_FOR_INIT_SIGNAL == False:
            # print("qqqq")
            pass
        time.sleep(2)
        WAIT_FOR_INIT_SIGNAL = False

        # print("wwww")

        if not EVAL_MODEL:
            time.sleep(5)
            print("开始发包")
            sp = os.popen('scp data/episode_data.pkl root@123.57.173.37:/root/data/0.pkl').read()
            print(sp)
            print("发包完成")
        