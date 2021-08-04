import cv2 as cv
import numpy as np

def img2dis(img):

    b, g, r = cv.split(img)
    color = ('b', 'g', 'r')  

    b_flatten = b.flatten()
    r_flatten = r.flatten()
    g_flatten = g.flatten()

    return np.array([b_flatten,r_flatten,g_flatten])