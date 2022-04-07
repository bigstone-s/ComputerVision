import numpy as np
import cv2
import matplotlib.pyplot as plt
from math import ceil, floor

# -------- 区域分裂合并算法 -------- #
# 1.分裂
#   将区域按田字形分裂成四个子区域
#   对于子区域，若其满足分裂条件，则继续分裂四个子区域
#   ... 以此递归，直到区域不满足分裂条件...
# 2.合并
#   对于分裂后的区域，若相邻区域满足相似条件，则将其合并，形成新的区域
#   ... 以此合并，直到每个区域与其相邻区域都不满足相似条件 ...

# ------------ 实现思路 ----------- #
# 1.分裂
#   使用递归分裂区域
#     递归时，需要考虑边界问题
#     具体而言，左下区域的长，右上区域的宽，右下区域的长和宽需要向上取整
# 2.合并
#   使用并查集完成合并操作
#   首先令图像的每个点自己形成一个集合
#   在回溯时，将所有分裂完得到的子区域按面积存在字典中
#   在回溯时，将所有分裂完得到的子区域中所有点连成一个集合
#   递归回溯完后，将字典中的区域按照面积从小到大排序
#   假设当前区域的左上角点为(x0,y0)，长和宽为w,h
#   (x0-1,y0), (x0+w,y0), (x0,y0-1), (x0,y0+h)为该区域相邻区域的点
#   通过并查集，可以通过相邻区域的点查找到相邻区域的信息
#   按照面积从小到大的顺序计算，可以确保所有相邻区域有交互
#   若当前区域与其相邻区域满足相似条件，则将当前区域的集合并入相邻区域的集合
#   由此完成后，得到的每一个集合储存一个图像分割对应的区域
# 3.其他
#   用像素均值度量区域相似度
#   用像素值减去均值后小于等于标准差的像素所占比例判断区域是否继续分裂


# ---- 超参数 ---- #
# 使用方法一判断分裂条件时的比率阈值
RATE = 0.98
# 使用方法二判断分裂条件时的最大最小值差的阈值
MAX_MIN = 1
# 使用均值判断区域相似度的阈值
MEAN = 5

def split_con(img, x, y, w, h):
    # 是否需要继续split
    # 方法一：
    # sub = img[x: x + w, y: y + h]
    # cnt = len(np.where(np.abs(sub - np.mean(sub)) <= np.std(sub))[1])
    # return cnt / (w * h) < RATE
    # 方法二：使用最大值最小值的差判断
    sub = img[x: x + w, y: y + h]
    return np.abs(np.max(sub) - np.min(sub)) > MAX_MIN

def find(A, p):
    # 包含路径压缩作用
    # 让所有节点（对应于一个区域）直接指向其祖宗节点
    if p[A] != A: p[A] = find(p[A], p)
    return p[A]

def split_merge(img, x, y, w, h, st, p, node_num, node_val, dic):
    '''
    :param img: 进行分割的图像
    :param x: 当前区域左上角点的x坐标
    :param y: 当前区域左上角点的y坐标
    :param w: 当前区域的宽
    :param h: 当前区域的高
    :param st: 判断区域是否被计算过
    :param p: 储存所有区域的父节点
    :param node_num: 储存每个区域的像素点数量
    :param node_val: 储存每个区域的像素值之和（与node_num结合算区域均值）
    :param dic: 存储划分好的区域
    :return: 划分好的区域（dic）
    '''
    # 递归时，需要注意区域上下取整的问题
    # 分裂时，对于左下角区域的长w/2、右上角区域的宽h/2、右下角区域的长和宽w/2 h/2，需要向上取整
    if split_con(img, x, y, w, h) and w > 1 and h > 1:
        split_merge(img, x, y, w // 2, h // 2, st, p, node_num, node_val, dic)
        split_merge(img, x + w // 2, y, ceil(w / 2), h // 2, st, p, node_num, node_val, dic)
        split_merge(img, x, y + h // 2, w // 2, ceil(h / 2), st, p, node_num, node_val, dic)
        split_merge(img, x + w // 2, y + h // 2, ceil(w / 2), ceil(h / 2), st, p, node_num, node_val, dic)

    if np.sum(st[x: x + w, y: y + h]) == 0:
        # 标记该区域被拓展
        st[x: x + w, y: y + h] = 1
        # 将区域按面积分类放入字典dic，同一面积的区域用list储存
        if w * h not in dic:
            dic[w * h] = []
        dic[w * h].append((x, y, w, h))
        # 将区域内所有点加入相同集合
        # 此处让所有点的祖宗节点指向左上角的点（通过路径压缩后，即父节点指向祖宗节点）
        for i in range(x, x + w):
            for j in range(y, y + h):
                if i != x or j != y:
                    xy, ij = find((x, y), p), find((i, j), p)
                    p[ij] = xy
                    # 同时将node_num和node_val也并入左上角的点
                    node_num[xy] += node_num[ij]
                    node_val[xy] += node_val[ij]

def model(img):
    '''img是灰度图'''
    region = np.zeros_like(img)
    # 划分的回溯时，判断区域是否被拓展
    st = np.zeros_like(img)
    W, H = img.shape
    dic, id_val = dict(), 1
    # 存储每个点（对应于区域）的父节点、以当前点为祖宗节点的集合的像素点数量
    # 以当前点为祖宗节点的集合的像素点值之和、以当前点为祖宗节点的集合中所有区域所属的类别
    p, node_num, node_val, id = {}, {}, {}, {}
    # 初始化，令所有像素点的父节点指向自己，即每个像素点都是一个集合
    for i in range(W):
        for j in range(H):
            p[(i, j)] = (i, j)
            node_num[(i, j)] = 1
            node_val[(i, j)] = int(img[i, j])
            id[(i, j)] = id_val; id_val += 1

    split_merge(img, 0, 0, W, H, st, p, node_num, node_val, dic)

    # 将划分的区域按面积由小到大排序
    dic = sorted(dic.items())
    # 由小到大遍历所有区域
    for s, reg_set in dic:
        for reg in reg_set:
            x, y, w, h = reg
            dx, dy = [-1, w + 1, 0, 0], [0, 0, -1, h + 1]
            # 拓展当前区域的上下左右四个区域
            # 从小区域到大区域，可以确保当前的拓展法能让所有相邻区域做到交互
            for i in range(4):
                x_near, y_near = x + dx[i], y + dy[i]
                # 判断拓展的点是否合法
                if (x_near < 0 or x_near >= W or y_near < 0 or y_near >= H):
                    continue
                # 找到当前区域和相邻区域的祖先节点
                curr = find((x, y), p)
                near = find((x_near, y_near), p)
                # 若两个区域的特征相似，则将其合并成一个集合（此处用均值判断特征相似度）
                curr_avg = node_val[curr] / node_num[curr]
                near_avg = node_val[near] / node_num[near]
                if np.abs(curr_avg - near_avg) < MEAN:
                    # 将当前节点的祖先节点指向其相邻节点的祖先节点
                    # 由于当前区域的节点都指向当前区域的祖先节点
                    # 相邻区域的节点都指向相邻区域的祖先节点
                    # 故此操作将两个区域合并成一个区域
                    p[curr] = near

    # 将每个点的类别写入region
    for i in range(W):
        for j in range(H):
            region[i, j] = id[find((i, j), p)]

    plt.imshow(region)
    plt.show()


if __name__ == '__main__':
    img = cv2.imread('../../images/shapes.png')
    # 输入函数的是灰度图
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    model(img)