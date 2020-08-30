# coding: utf-8
# This script is modified from https://github.com/lars76/kmeans-anchor-boxes

from __future__ import division, print_function

import numpy as np

def iou(box, clusters):
    """
    Calculates the Intersection over Union (IoU) between a box and k clusters.
    param:
        box: tuple or array, shifted to the origin (i. e. width and height)
        clusters: numpy array of shape (k, 2) where k is the number of clusters
    return:
        numpy array of shape (k, 0) where k is the number of clusters
    """
    #x和y是宽度和高度
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    #防止box和cluster出现为空的情况
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")
    #算的是面积，交集
    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]
    #返回浮点结果不做截断，防止完全遇上一样的box，因为就是聚类出来的，有可能遇到一样的为1
    iou_ = np.true_divide(intersection, box_area + cluster_area - intersection + 1e-10)
    # iou_ = intersection / (box_area + cluster_area - intersection + 1e-10)
    #返回以IOU
    return iou_


def avg_iou(boxes, clusters):
    """
    Calculates the average Intersection over Union (IoU) between a numpy array of boxes and k clusters.
    param:
        boxes: numpy array of shape (r, 2), where r is the number of rows
        clusters: numpy array of shape (k, 2) where k is the number of clusters
    return:
        average IoU as a single float
    """
    #clusters是聚类出来的，相当于中心点和标准点
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])


def translate_boxes(boxes):
    """
    Translates all the boxes to the origin.
    param:
        boxes: numpy array of shape (r, 4)
    return:
    numpy array of shape (r, 2)
    """
    #因为给的文件里面给的是xmin，ymin，xmax。ymax，所以要转换成宽度和高度，方便计算IOU
    new_boxes = boxes.copy()
    for row in range(new_boxes.shape[0]):
        new_boxes[row][2] = np.abs(new_boxes[row][2] - new_boxes[row][0])
        new_boxes[row][3] = np.abs(new_boxes[row][3] - new_boxes[row][1])
        #覆写，移除不要的[0,1]
    return np.delete(new_boxes, [0, 1], axis=1)


def kmeans(boxes, k, dist=np.median):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    param:
        boxes: numpy array of shape (r, 2), where r is the number of rows
        k: number of clusters
        dist: distance function
    return:
        numpy array of shape (k, 2)
    """
    rows = boxes.shape[0]

    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))

    np.random.seed()

    # the Forgy method will fail if the whole array contains the same rows
    #从rows中随机选择k个做为cluster，相当于K-means随机产生k个聚类,且不可以找相同数字
    clusters = boxes[np.random.choice(rows, k, replace=False)]

    while True:
        #rows个box和k个clusters之间的d
        #每个不同的box会得到k个值
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)
        #返回的是disatnce最小的那个的索引值，每一个框属于哪个聚类中心，调整这些框的归属
        nearest_clusters = np.argmin(distances, axis=1)

        if (last_clusters == nearest_clusters).all():#所有的返回值都为True才会执行，即当每个框属于某个聚类中心的索引不再更新时跳出循环
            break

        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)
            #dist=np.median,在列的方向上求中位数，
            #找到每一个cluster对应的那些框的分界点，分界点一边是属于这个cluster，调整聚类中心的位置

        last_clusters = nearest_clusters
        #返回的是每一个聚类中心重新计算中位数，反复迭代计算后的新聚类中心点

    return clusters

#直接从txt中每一行手动分离图片的size，box的size等参数
def parse_anno(annotation_path, target_size=None):
    anno = open(annotation_path, 'r')
    result = []
    for line in anno:
        s = line.strip().split(' ')
        img_w = int(s[2])
        img_h = int(s[3])
        s = s[4:]
        box_cnt = len(s) // 5
        for i in range(box_cnt):
            x_min, y_min, x_max, y_max = float(s[i*5+1]), float(s[i*5+2]), float(s[i*5+3]), float(s[i*5+4])
            width = x_max - x_min
            height = y_max - y_min
            #断言width和height为0，否则报错
            assert width > 0
            assert height > 0
            # use letterbox resize, i.e. keep the original aspect ratio
            # get k-means anchors on the resized target image size
            #等比例缩放
            if target_size is not None:
                resize_ratio = min(target_size[0] / img_w, target_size[1] / img_h)
                width *= resize_ratio
                height *= resize_ratio
                result.append([width, height])
            # get k-means anchors on the original image size
            else:
                result.append([width, height])
    result = np.asarray(result)
    #返回值是一个图片里面所有的box 不带label
    return result


def get_kmeans(anno, cluster_num=9):

    anchors = kmeans(anno, cluster_num)
    ave_iou = avg_iou(anno, anchors)

    anchors = anchors.astype('int').tolist()
    #按照大小排序
    anchors = sorted(anchors, key=lambda x: x[0] * x[1])

    return anchors, ave_iou


if __name__ == '__main__':
    # target resize format: [width, height]
    # if target_resize is speficied, the anchors are on the resized image scale
    # if target_resize is set to None, the anchors are on the original image scale
    target_size = [416, 416]
    annotation_path = "./misc/experiments_on_voc/val.txt"
    anno_result = parse_anno(annotation_path, target_size=target_size)
    anchors, ave_iou = get_kmeans(anno_result, 9)

    anchor_string = ''
    for anchor in anchors:
        anchor_string += '{},{}, '.format(anchor[0], anchor[1])
    anchor_string = anchor_string[:-2]

    print('anchors are:')
    print(anchor_string)
    print('the average iou is:')
    print(ave_iou)

