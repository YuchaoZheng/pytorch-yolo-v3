from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2
import matplotlib.pyplot as plt
from bbox import bbox_iou


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def count_learnable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def convert2cpu(matrix):
    if matrix.is_cuda:
        return torch.FloatTensor(matrix.size()).copy_(matrix)
    else:
        return matrix


"""
在特征图上进行多尺度预测, 在GRID每个位置都有三个不同尺度的锚点.predict_transform()利用一个scale得到的feature map预测得到的每个anchor的属性(x,y,w,h,s,s_cls1,s_cls2...),其中x,y,w,h
是在网络输入图片坐标系下的值,s是方框含有目标的置信度得分，s_cls1,s_cls_2等是方框所含目标对应每类的概率。输入的feature map(prediction变量) 
维度为(batch_size, num_anchors * bbox_attrs, grid_size, grid_size)，类似于一个batch彩色图片BxCxHxW存储方式。参数见predict_transform()里面的变量。
并且将结果的维度变换成(batch_size, grid_size*grid_size*num_anchors, 5+类别数量)的tensor，同时得到每个方框在网络输入图片(416x416)坐标系下的(x,y,w,h)以及方框含有目标的得分以及每个类的得分。
"""


def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA=True):
    batch_size = prediction.size(0)
    # stride表示的是整个网络的步长，等于图像原始尺寸与yolo层输入的feature mapr尺寸相除，因为输入图像是正方形，所以用高相除即可
    stride = inp_dim // prediction.size(2)  # 416//13=32
    # feature map每条边格子的数量，416//32=13
    grid_size = inp_dim // stride
    # 一个方框属性个数，等于5+类别数量
    bbox_attrs = 5 + num_classes
    # anchors数量
    num_anchors = len(anchors)

    # 每一个都对应到feature map的尺寸
    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]

    # 输入的prediction维度为(batch_size, num_anchors * bbox_attrs, grid_size, grid_size)，类似于一个batch彩色图片BxCxHxW
    # 存储方式，将它的维度变换成(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)

    # Sigmoid the  centre_X, centre_Y. and object confidencce
    # Sigmoid the tX, tY. and object confidencce.tx与ty为预测的坐标偏移值
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

    # Add the center offsets
    # 这里生成了每个格子的左上角坐标，生成的坐标为grid x grid的二维数组，a，b分别对应这个二维矩阵的x,y坐标的数组，a,b的维度与grid维度一样。
    # 每个grid cell的尺寸均为1，故grid范围是[0,12]（假如当前的特征图13*13）
    grid_len = np.arange(grid_size)
    a, b = np.meshgrid(grid_len, grid_len)

    # x_offset即cx,y_offset即cy，表示当前cell左上角坐标
    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    # 这里的x_y_offset对应的是最终的feature map中每个格子的左上角坐标，比如有13个格子，刚x_y_offset的坐标就对应为(0,0),(0,1)…(12,12) .view(-1, 2)将tensor变成两列，unsqueeze(0)在0维上添加了一维。
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)

    # bx=sigmoid(tx)+cx,by=sigmoid(ty)+cy
    prediction[:, :, :2] += x_y_offset

    # log space transform height and the width
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    # 公式bw=pw×e^tw及bh=ph×e^th，pw为anchorbox的长度
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors

    # Softmax the class scores
    prediction[:, :, 5: 5 + num_classes] = torch.sigmoid((prediction[:, :, 5: 5 + num_classes]))

    # 将相对于最终feature map的方框坐标和尺寸映射回输入网络图片(416x416)，即将方框的坐标乘以网络的stride即可
    prediction[:, :, :4] *= stride

    return prediction


def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names


def get_im_dim(im):
    im = cv2.imread(im)
    w, h = im.shape[1], im.shape[0]
    return w, h


# 因为同一类别可能会有多个真实检测结果，所以我们使用unique函数来去除重复的元素，即一类只留下一个元素，达到获取任意给定图像中存在的类别的目的。
def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    # np.unique该函数是去除数组中的重复数字，并进行排序之后输出
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    # 复制数据
    tensor_res = tensor.new(unique_tensor.shape)
    # new(args, *kwargs) 构建[相同数据类型]的新Tensor
    tensor_res.copy_(unique_tensor)
    return tensor_res


# 形状为 B x 10647 x 85 的张量；其中 B 是指一批（batch）中图像的数量，10647 是每个图像中所预测的边界框的数量，85 是指边界框属性的数量
# 输入为预测结果、置信度（objectness 分数阈值）、num_classes（我们这里是 80）和 nms_conf（NMS IoU 阈值）
'''
必须使我们的输出满足 objectness 分数阈值和非极大值抑制（NMS），以得到后文所提到的「真实」检测结果。要做到这一点就要用 write_results函数。
函数的输入为预测结果、置信度（objectness 分数阈值）、num_classes（我们这里是 80）和 nms_conf（NMS IoU 阈值）。
write_results()首先将网络输出方框属性(x,y,w,h)转换为在网络输入图片(416x416)坐标系中，方框左上角与右下角坐标(x1,y1,x2,y2)，以方便NMS操作。
然后将方框含有目标得分低于阈值的方框去掉，提取得分最高的那个类的得分max_conf，同时返回这个类对应的序号max_conf_score,
然后进行NMS操作。最终每个方框的属性为(ind,x1,y1,x2,y2,s,s_cls,index_cls)，ind 是这个方框所属图片在这个batch中的序号，
x1,y1是在网络输入图片(416x416)坐标系中，方框左上角的坐标；x2,y2是在网络输入图片(416x416)坐标系中，方框右下角的坐标。
s是这个方框含有目标的得分,s_cls是这个方框中所含目标最有可能的类别的概率得分，index_cls是s_cls对应的这个类别所对应的序号.
'''
def write_results(prediction, confidence, num_classes, nms=True, nms_conf=0.4):
    # confidence: 输入的预测shape=(B,10647, 85)。conf_mask: shape=(B,10647) => 增加一维度之后 (B, 10647, 1)
    # 我们的预测张量包含有关Bx10647边界框的信息。对于含有目标的得分小于confidence的每个方框，它对应的含有目标的得分将变成0,即conf_mask中对应元素为0.而保留预测结果中置信度大于给定阈值的部分prediction的conf_mask
    conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
    # 根据numpy的广播原理，它会扩展成与prediction维度一样的tensor，所以含有目标的得分小于confidence的方框所有的属性都会变为0，故如果没有检测任何有效目标,则返回值为0
    prediction = prediction * conf_mask

    try:
        ind_nz = torch.nonzero(prediction[:, :, 4]).transpose(0, 1).contiguous()
    except:
        return 0

    '''
    保留预测结果中置信度大于阈值的bbox
    下面开始为nms准备
    '''
    # prediction的前五个数据分别表示 (Cx, Cy, w, h, score)，这里创建一个新的数组，大小与predicton的大小相同
    box_a = prediction.new(prediction.shape)
    '''
    我们可以将我们的框的 (中心 x, 中心 y, 高度, 宽度) 属性转换成 (左上角 x, 左上角 y, 右下角 x, 右下角 y)
    这样做用每个框的两个对角坐标能更轻松地计算两个框的 IoU
    '''
    box_a[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2] / 2)
    box_a[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3] / 2)
    box_a[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2] / 2)
    box_a[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3] / 2)
    # 计算后的新坐标复制回去
    prediction[:, :, :4] = box_a[:, :, :4]

    # 每张图像中的「真实」检测结果的数量可能存在差异。比如，一个大小为 3 的 batch 中有 1、2、3 这 3 张图像，它们各自有 5、2、4 个「真实」检测结果。
    # 因此，一次只能完成一张图像的置信度阈值设置和 NMS。
    # 也就是说，我们不能将所涉及的操作向量化，而且必须在预测的第一个维度（包含一个 batch 中图像的索引）上循环。

    # 第0个维度是batch_size
    batch_size = prediction.size(0)

    # shape=(1,85+1)
    output = prediction.new(1, prediction.size(2) + 1)
    # 拼接结果到output中最后返回
    write = False

    # 对每一张图片得分的预测值进行NMS操作，因为每张图片的目标数量不一样，所以有效得分的方框的数量不一样，没法将几张图片同时处理，因此一次只能完成一张图的置信度阈值的设置和NMS,不能将所涉及的操作向量化.
    # 所以必须在预测的第一个维度上（batch数量）上遍历每张图片，将得分低于一定分数的去掉，对剩下的方框进行进行NMS
    for ind in range(batch_size):
        # select the image from the batch
        # 选择此batch中第ind个图像的预测结果,image_pred对应一张图片中所有方框的坐标(x1,y1,x2,y2)以及得分，是一个二维tensor 维度为10647x85
        image_pred = prediction[ind]

        # Get the class having maximum score, and the index of that class
        # Get rid of num_classes softmax scores
        # Add the class index and the class score of class having maximum score
        # 注意每个边界框行都有 85 个属性，其中 80 个是类别分数。此时，我们只关心有最大值的类别分数。
        # 所以，我们移除了每一行的这 80 个类别分数，并且转而增加了有最大值的类别的索引以及那一类别的类别分数。
        # 我们只关心有最大值的类别分数，prediction[:, 5:]表示每一分类的分数,返回每一行中所有类别的得分最高的那个类的得分max_conf，同时返回这个类对应的序号max_conf_score
        # 维度扩展max_conf: shape=(10647->15) => (10647->15,1)添加一个列的维度，max_conf变成二维tensor，尺寸为10647x1
        max_conf, max_conf_score = torch.max(image_pred[:, 5:5 + num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:, :5], max_conf, max_conf_score)
        # shape=(10647, 5+1+1=7)
        image_pred = torch.cat(seq, 1)

        # image_pred[:,4]是长度为10647的一维tensor,维度为4的列是置信度分数。假设有15个框含有目标的得分非0，返回15x1的tensor
        non_zero_ind = (torch.nonzero(image_pred[:, 4]))

        image_pred_ = image_pred[non_zero_ind.squeeze(), :].view(-1, 7)

        # Get the various classes detected in the image
        try:
            # 获取一张图像中所检测到的类别,因为同一类别可能会有多个「真实」检测结果，所以我们使用一个名叫 unique 的函数来获取任意给定图像中存在的类别。
            img_classes = unique(image_pred_[:, -1])
        except:
            continue
        # WE will do NMS classwise
        # 按照类别执行 NMS
        for cls in img_classes:
            # get the detections with one particular class
            cls_mask = image_pred_ * (image_pred_[:, -1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:, -2]).squeeze()

            image_pred_class = image_pred_[class_mask_ind].view(-1, 7)

            # sort the detections such that the entry with the maximum objectness
            # confidence is at the top

            # 开始 nms
            # 按照score排序, 由大到小
            # 最大值最上面

            conf_sort_index = torch.sort(image_pred_class[:, 4], descending=True)[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)

            # if nms has to be done
            if nms:
                # For each detection
                for i in range(idx):
                    # Get the IOUs of all boxes that come after the one we are looking at
                    # in the loop
                    try:
                        ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i + 1:])
                    except ValueError:
                        break

                    except IndexError:
                        break

                    # Zero out all the detections that have IoU > treshhold
                    iou_mask = (ious < nms_conf).float().unsqueeze(1)
                    image_pred_class[i + 1:] *= iou_mask

                    # Remove the non-zero entries
                    non_zero_ind = torch.nonzero(image_pred_class[:, 4]).squeeze()
                    image_pred_class = image_pred_class[non_zero_ind].view(-1, 7)

            # Concatenate the batch_id of the image to the detection
            # this helps us identify which image does the detection correspond to
            # We use a linear straucture to hold ALL the detections from the batch
            # the batch_dim is flattened
            # batch is identified by extra batch column

            # write_results 函数输出一个形状为 Dx8 的张量；其中 D 是所有图像中的「真实」检测结果，每个都用一行表示。
            # 每一个检测结果都有 8 个属性，即：该检测结果所属的 batch 中图像的索引、4 个角的坐标、objectness 分数、有最大置信度的类别的分数、
            # 该类别的索引
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
            seq = batch_ind, image_pred_class
            if not write:
                output = torch.cat(seq, 1)
                write = True
            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output, out))

    return output


# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 00:12:16 2018

@author: ayooshmac
"""


def predict_transform_half(prediction, inp_dim, anchors, num_classes, CUDA=True):
    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2)

    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)
    grid_size = inp_dim // stride

    prediction = prediction.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)

    # Sigmoid the  centre_X, centre_Y. and object confidencce
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

    # Add the center offsets
    grid_len = np.arange(grid_size)
    a, b = np.meshgrid(grid_len, grid_len)

    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)

    if CUDA:
        x_offset = x_offset.cuda().half()
        y_offset = y_offset.cuda().half()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)

    prediction[:, :, :2] += x_y_offset

    # log space transform height and the width
    anchors = torch.HalfTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors

    # Softmax the class scores
    prediction[:, :, 5: 5 + num_classes] = nn.Softmax(-1)(Variable(prediction[:, :, 5: 5 + num_classes])).data

    prediction[:, :, :4] *= stride

    return prediction


def write_results_half(prediction, confidence, num_classes, nms=True, nms_conf=0.4):
    conf_mask = (prediction[:, :, 4] > confidence).half().unsqueeze(2)
    prediction = prediction * conf_mask

    try:
        ind_nz = torch.nonzero(prediction[:, :, 4]).transpose(0, 1).contiguous()
    except:
        return 0

    box_a = prediction.new(prediction.shape)
    box_a[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2] / 2)
    box_a[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3] / 2)
    box_a[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2] / 2)
    box_a[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3] / 2)
    prediction[:, :, :4] = box_a[:, :, :4]

    batch_size = prediction.size(0)

    output = prediction.new(1, prediction.size(2) + 1)
    write = False

    for ind in range(batch_size):
        # select the image from the batch
        image_pred = prediction[ind]

        # Get the class having maximum score, and the index of that class
        # Get rid of num_classes softmax scores
        # Add the class index and the class score of class having maximum score
        max_conf, max_conf_score = torch.max(image_pred[:, 5:5 + num_classes], 1)
        max_conf = max_conf.half().unsqueeze(1)
        max_conf_score = max_conf_score.half().unsqueeze(1)
        seq = (image_pred[:, :5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)

        # Get rid of the zero entries
        non_zero_ind = (torch.nonzero(image_pred[:, 4]))
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(), :]
        except:
            continue

        # Get the various classes detected in the image
        img_classes = unique(image_pred_[:, -1].long()).half()

        # WE will do NMS classwise
        for cls in img_classes:
            # get the detections with one particular class
            cls_mask = image_pred_ * (image_pred_[:, -1] == cls).half().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:, -2]).squeeze()

            image_pred_class = image_pred_[class_mask_ind]

            # sort the detections such that the entry with the maximum objectness
            # confidence is at the top
            conf_sort_index = torch.sort(image_pred_class[:, 4], descending=True)[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)

            # if nms has to be done
            if nms:
                # For each detection
                for i in range(idx):
                    # Get the IOUs of all boxes that come after the one we are looking at
                    # in the loop
                    try:
                        ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i + 1:])
                    except ValueError:
                        break

                    except IndexError:
                        break

                    # Zero out all the detections that have IoU > treshhold
                    iou_mask = (ious < nms_conf).half().unsqueeze(1)
                    image_pred_class[i + 1:] *= iou_mask

                    # Remove the non-zero entries
                    non_zero_ind = torch.nonzero(image_pred_class[:, 4]).squeeze()
                    image_pred_class = image_pred_class[non_zero_ind]

            # Concatenate the batch_id of the image to the detection
            # this helps us identify which image does the detection correspond to
            # We use a linear straucture to hold ALL the detections from the batch
            # the batch_dim is flattened
            # batch is identified by extra batch column
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
            seq = batch_ind, image_pred_class

            if not write:
                output = torch.cat(seq, 1)
                write = True
            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output, out))

    return output
