import threading
import os
import time
def getImg():
    while 1:
        ret = os.popen('adb shell screencap -p /sdcard/tmp.png').read()
        ret = os.popen('adb pull /sdcard/tmp.png').read()
        #print('截图正在回传...')

if __name__ == '__main__':
    t1 = threading.Thread(target=getImg)
    t1.setDaemon(True)
    t1.start()
    t1.join()
