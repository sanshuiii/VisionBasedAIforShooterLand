import os
import time

def getImg():
    curT = time.time()
    while 1:
        nxtT = time.time()
        if nxtT - curT >= 0.5:
            os.popen('adb shell screencap -p /sdcard/tmp.png').read()
            os.popen('adb pull /sdcard/tmp.png').read()
            curT = nxtT
        #print('截图正在回传...')

if __name__ == '__main__':
    pass
