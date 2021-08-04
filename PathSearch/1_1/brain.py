import numpy as np
import cv2

# 自己头上的绿色血条    [55,230,200] - [65,240,210]
# 怪物头上的红色血条    [3,240,178] - [10,255,255] 小怪
# 障碍物1 [5,182,100] - [45,230,235]
# 障碍物2 [0,0,0] - [35,60,235]

'''通过大面积蓝色确定进入技能选择界面'''
def is_select_skill(img):
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    low = [100,170,0]
    high = [255,255,255]
    mask = cv2.inRange(hsv, np.array(low), np.array(high))
    target = len(mask[mask==255])
    bg = len(mask[mask==0])
    # 还要判断是不是最后一关前的女神祝福
    if target/(target+bg) > 0.15:
        low = [0,240,240]
        high = [15,255,255]
        mask = cv2.inRange(hsv, np.array(low), np.array(high))[450:500,:]
        target = len(mask[mask==255])
        bg = len(mask[mask==0])
        if target/(target+bg) > 0.1:
            return True,True
        else:
            return True,False
    else:
        return False,False


'''通过头顶的红色血条定位怪兽'''
def is_under_attack(img):
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    low = [3,240,178]
    high = [10,255,255]
    mask = cv2.inRange(hsv, np.array(low), np.array(high))
    contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours):
        return True
    else:
        return False

'''寻路规划器'''
from copy import deepcopy
obstacle_dict = {
    'zhalan':[[5,182,100],[45,230,235]],
    'shizhu':[[0,0,0],[35,60,235]],
    'jianci':[[105,65,140],[115,255,255]],
}

class Planner:
    def __init__(self,max_length,img,w,h,debug=False):
        
        # 定位英雄坐标
        self.color_img = img
        self.origin = self.where_am_I(self.color_img)
        # 生成障碍物地图（二值）
        hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        mask = np.zeros((585,270),dtype=np.uint8)
        for low,high in obstacle_dict.values():
            tmp = cv2.inRange(hsv, np.array(low), np.array(high))
            mask = cv2.bitwise_or(mask,tmp)
        self.gray_img = mask
        self.gray_img_height,self.gray_img_width= self.gray_img.shape
        # 最大搜索步数
        self.max_length = max_length
        # 一次搜索步进长度
        assert w == h
        self.w = w
        self.h = h
        # 存储状态，无需反复判断
        self.connectivity_hashmap = {}
        # 可行路径集
        self.result = []
        # dfs时暂存的路径栈
        self.path = []
        # 是否开debug模式
        self.debug = debug

    def where_am_I(self,img):
        hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        low = [55,230,200]
        high = [65,240,210]
        mask = cv2.inRange(hsv, np.array(low), np.array(high))
        try:
            hero = np.mean(np.where(mask==255),axis=1).astype(np.int16)
        except:
            hero = np.array([img.shape[0]*0.5,img.shape[1]*0.5]).astype(np.int16)
        return tuple(hero)
    
    def check_connectivity(self,x,y,x_,y_):
        if self.connectivity_hashmap.get((x,y,x_,y_)) == None:
            if x == x_:
                x1 = self.origin[1] + x*self.w - 5
                x2 = self.origin[1] + x*self.w + 5
                y1 = min(self.origin[0] - y*self.h,self.origin[0] - y_*self.h)
                y2 = max(self.origin[0] - y*self.h,self.origin[0] - y_*self.h)
            else:
                x1 = min(self.origin[1] + x*self.w,self.origin[1] + x_*self.w)      +(x_-x)*10
                x2 = max(self.origin[1] + x*self.w,self.origin[1] + x_*self.w)      +(x_-x)*10
                y1 = self.origin[0] - y*self.h - 5                                  + 20
                y2 = self.origin[0] - y*self.h + 5                                  + 20
            
            if x1 < 0 or x2 > self.gray_img_width or y1 < 0 or y2 > self.gray_img_height:
                connectivity = False
            else:
                area = self.gray_img[y1:y2,x1:x2]
                obstacle= len(area[area==255])
                obstacle_ratio = obstacle / (10*self.w)
                #print(obstacle_ratio)
                connectivity = True if obstacle_ratio < 0.25 else False

            self.connectivity_hashmap[((x,y,x_,y_))]=connectivity
            self.connectivity_hashmap[((x_,y_,x,y))]=connectivity
        
            if self.debug:
                if connectivity:
                    cv2.rectangle(self.color_img, (x1,y1), (x2,y2), (0,255,0), 1)
                else:
                    cv2.rectangle(self.color_img, (x1,y1), (x2,y2), (0,0,255), 1)
                cv2.imshow('img',self.color_img)
                cv2.waitKey(0)
        
        return self.connectivity_hashmap.get((x,y,x_,y_))

    def dfs(self,x,y):
        self.path.append((x,y))
        if len(self.path) == self.max_length:
            self.result.append(deepcopy(self.path))
            self.path.pop()
            return
        
        forward_connectivity = self.check_connectivity(x,y,x,y+1)
        left_connectivity = False if (x-1,y) in self.path else self.check_connectivity(x,y,x-1,y)
        right_connectivity = False if (x+1,y) in self.path else self.check_connectivity(x,y,x+1,y)

        if forward_connectivity==False and left_connectivity==False and right_connectivity==False:
            self.result.append(deepcopy(self.path))
            self.path.pop()
            return
        
        if forward_connectivity:
            self.dfs(x,y+1)
        
        if left_connectivity:
            self.dfs(x-1,y)

        if right_connectivity:
            self.dfs(x+1,y)
        
        self.path.pop()
        return
    
    def get_result(self):
        self.dfs(0,0)
        ret = []
        for path in self.result:
            score = 0.
            action_list = []
            for i in range(1,len(path)):
                if path[i][1] - path[i-1][1] == 1:
                    action_list.append(1)
                    score += 1
                elif path[i][0] - path[i-1][0] == 1:
                    action_list.append(3)
                    score -= 0.2
                else:
                    action_list.append(2)
                    score -= 0.2
            ret.append((action_list,score))
        
        if len(ret) == 1:
            return [4,]

        ret = sorted(ret,key= lambda x : x[1],reverse=True)[0][0]
        while len(ret)> 0 and ret[-1] != 1:
            ret.pop()
        
        if len(ret) > 2: return ret[:2]
        if len(ret) < 1: return [4,]
        return ret

if __name__ == '__main__':
    img = cv2.imread('tmp.png')
    img = cv2.resize(img,(270,585))
    planner = Planner(max_length=5,img=img,w=25,h=25,debug=1)
    res = planner.get_result()
    print(res)

    # img = cv2.imread('tmp.png')
    # is_select_skill(img)
    
    # from pathlib import Path
    # jpg_files = Path('./2').glob("*.png")

    # for jpg_file in jpg_files:
    #     img = cv2.imread(str(jpg_file))
    #     mask = search_road(img,debug=True)
    #     fn = 'res/'+jpg_file.stem+'.png'
    #     cv2.imwrite(fn,mask)
