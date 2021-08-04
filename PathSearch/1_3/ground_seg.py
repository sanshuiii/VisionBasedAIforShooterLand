import cv2 as cv
import numpy as np

def to_img_binary(img):
    """
    :param img:传入一张要分割的图片
    :return: 返回二值图片
    """
    # img = cv.imread('./2/2021-07-27-09-53-26-025.png')
    # print(img.shape)
    # 截取roi
    img_roi = img[200:300, 80:190]
    # print(img_roi.shape)
    b, g, r = cv.split(img_roi)
    b_full, g_full, r_full = cv.split(img)
    color = ('b', 'g', 'r')  # 稍微调整显示颜色，提高可视化效果

    #  颜色直方图=
    # for id, bgrcolor in enumerate(color):
    #    sns.kdeplot(img[:, :, id].flatten(), shade=True, color=bgrcolor, label=bgrcolor, alpha=.7)
    #    plt.title('Histrogram of Color image')
    # plt.show()

    N = 2
    # 分成三个通道统计
    b_flatten = b.flatten()
    # b_max = np.argmax(np.bincount(b_flatten)) 计算最大值
    # print(b_max)
    b_sigma = N * np.std(b_flatten)
    b_u = np.mean(b_flatten)
    b_up = b_u + b_sigma
    b_down = b_u - b_sigma

    r_flatten = r.flatten()
    r_sigma = N * np.std(r_flatten)
    r_u = np.mean(r_flatten)
    r_up = r_u + r_sigma
    r_down = r_u - r_sigma

    g_flatten = g.flatten()
    g_sigma = N * np.std(g_flatten)
    g_u = np.mean(g_flatten)
    g_up = g_u + g_sigma
    g_down = g_u - g_sigma


    binary_img = np.zeros([img.shape[0], img.shape[1]])
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if b_up > img[i, j, 0] > b_down and g_up > img[i, j, 1] > g_down and r_up > img[i, j, 2] > r_down:
                # 在阈值范围内
                binary_img[i, j] = 0
            else:
                binary_img[i, j] = 255

    return binary_img

import cv2
if __name__ == '__main__':
    img = cv2.imread('2\\2021-08-03-17-02-39-996.png')
    mask = to_img_binary(img)
    cv2.imwrite('mask.png',mask)