import argparse
import os.path
from mindspore import load_checkpoint, load_param_into_net, context
import models


"""Eval Retinaface_resnet50."""
import time
import datetime
import cv2
import math
import mindspore as ms
from mindspore.common import set_seed
from itertools import product
from src.config import cfg_res50
from src.network import RetinaFace, resnet50
from src.utils import decode_bbox, prior_box
import json
from mindspore import Tensor
from PIL import Image
import matplotlib.pyplot as plt
import mindspore.dataset as ds
import numpy as np
import os
from src01.models import define_net, load_ckpt
from src01.model_utils.config import config
import mindspore.ops as ops

set_seed(1)



class Timer():
    def __init__(self):
        self.start_time = 0.
        self.diff = 0.

    def start(self):
        self.start_time = time.time()

    def end(self):
        self.diff = time.time() - self.start_time

class DetectionEngine:
    def __init__(self, cfg):
        self.results = {}
        self.nms_thresh = cfg['val_nms_threshold']
        self.conf_thresh = cfg['val_confidence_threshold']
        self.iou_thresh = cfg['val_iou_threshold']
        self.var = cfg['variance']
        self.save_prefix = cfg['val_predict_save_folder']
        self.gt_dir = cfg['val_gt_dir']

    def _iou(self, a, b):
        A = a.shape[0]
        B = b.shape[0]
        max_xy = np.minimum(
            np.broadcast_to(np.expand_dims(a[:, 2:4], 1), [A, B, 2]),
            np.broadcast_to(np.expand_dims(b[:, 2:4], 0), [A, B, 2]))
        min_xy = np.maximum(
            np.broadcast_to(np.expand_dims(a[:, 0:2], 1), [A, B, 2]),
            np.broadcast_to(np.expand_dims(b[:, 0:2], 0), [A, B, 2]))
        inter = np.maximum((max_xy - min_xy + 1), np.zeros_like(max_xy - min_xy))
        inter = inter[:, :, 0] * inter[:, :, 1]

        area_a = np.broadcast_to(
            np.expand_dims(
                (a[:, 2] - a[:, 0] + 1) * (a[:, 3] - a[:, 1] + 1), 1),
            np.shape(inter))
        area_b = np.broadcast_to(
            np.expand_dims(
                (b[:, 2] - b[:, 0] + 1) * (b[:, 3] - b[:, 1] + 1), 0),
            np.shape(inter))
        union = area_a + area_b - inter
        return inter / union

    def _nms(self, boxes, threshold=0.5):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        scores = boxes[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        reserved_boxes = []
        while order.size > 0:
            i = order[0]
            reserved_boxes.append(i)
            max_x1 = np.maximum(x1[i], x1[order[1:]])
            max_y1 = np.maximum(y1[i], y1[order[1:]])
            min_x2 = np.minimum(x2[i], x2[order[1:]])
            min_y2 = np.minimum(y2[i], y2[order[1:]])

            intersect_w = np.maximum(0.0, min_x2 - max_x1 + 1)
            intersect_h = np.maximum(0.0, min_y2 - max_y1 + 1)
            intersect_area = intersect_w * intersect_h
            ovr = intersect_area / (areas[i] + areas[order[1:]] - intersect_area)

            indices = np.where(ovr <= threshold)[0]
            order = order[indices + 1]

        return reserved_boxes

    def write_result(self):
        # save result to file.
        import json
        t = datetime.datetime.now().strftime('_%Y_%m_%d_%H_%M_%S')
        try:
            if not os.path.isdir(self.save_prefix):
                os.makedirs(self.save_prefix)

            self.file_path = self.save_prefix + '/predict' + t + '.json'
            f = open(self.file_path, 'w')
            json.dump(self.results, f)
        except IOError as e:
            raise RuntimeError("Unable to open json file to dump. What(): {}".format(str(e)))
        else:
            f.close()
            return self.file_path

    def detect(self, boxes, confs, resize, scale, image_path, priors):
        if boxes.shape[0] == 0:
            # add to result
            event_name, img_name = image_path.split('/')
            self.results[event_name][img_name[:-4]] = {'img_path': image_path,
                                                       'bboxes': []}
            return

        boxes = decode_bbox(np.squeeze(boxes.asnumpy(), 0), priors, self.var)
        boxes = boxes * scale / resize

        scores = np.squeeze(confs.asnumpy(), 0)[:, 1]
        # ignore low scores
        inds = np.where(scores > self.conf_thresh)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1]
        boxes = boxes[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = self._nms(dets, self.nms_thresh)
        dets = dets[keep, :]

        dets[:, 2:4] = (dets[:, 2:4].astype(np.int) - dets[:, 0:2].astype(np.int)).astype(np.float) # int
        dets[:, 0:4] = dets[:, 0:4].astype(np.int).astype(np.float)                                 # int


        # add to result
        event_name, img_name = image_path.split('/')
        if event_name not in self.results.keys():
            self.results[event_name] = {}
        self.results[event_name][img_name[:-4]] = {'img_path': image_path,
                                                   'bboxes': dets[:, :5].astype(np.float).tolist()}

    def _get_gt_boxes(self):
        from scipy.io import loadmat
        gt = loadmat(os.path.join(self.gt_dir, 'wider_face_val.mat'))
        hard = loadmat(os.path.join(self.gt_dir, 'wider_hard_val.mat'))
        medium = loadmat(os.path.join(self.gt_dir, 'wider_medium_val.mat'))
        easy = loadmat(os.path.join(self.gt_dir, 'wider_easy_val.mat'))

        faceboxes = gt['face_bbx_list']
        events = gt['event_list']
        files = gt['file_list']

        hard_gt_list = hard['gt_list']
        medium_gt_list = medium['gt_list']
        easy_gt_list = easy['gt_list']

        return faceboxes, events, files, hard_gt_list, medium_gt_list, easy_gt_list

    def _norm_pre_score(self):
        max_score = 0
        min_score = 1

        for event in self.results:
            for name in self.results[event].keys():
                bbox = np.array(self.results[event][name]['bboxes']).astype(np.float)
                if bbox.shape[0] <= 0:
                    continue
                max_score = max(max_score, np.max(bbox[:, -1]))
                min_score = min(min_score, np.min(bbox[:, -1]))

        length = max_score - min_score
        for event in self.results:
            for name in self.results[event].keys():
                bbox = np.array(self.results[event][name]['bboxes']).astype(np.float)
                if bbox.shape[0] <= 0:
                    continue
                bbox[:, -1] -= min_score
                bbox[:, -1] /= length
                self.results[event][name]['bboxes'] = bbox.tolist()

    def _image_eval(self, predict, gt, keep, iou_thresh, section_num):

        _predict = predict.copy()
        _gt = gt.copy()

        image_p_right = np.zeros(_predict.shape[0])
        image_gt_right = np.zeros(_gt.shape[0])
        proposal = np.ones(_predict.shape[0])

        # x1y1wh -> x1y1x2y2
        _predict[:, 2:4] = _predict[:, 0:2] + _predict[:, 2:4]
        _gt[:, 2:4] = _gt[:, 0:2] + _gt[:, 2:4]

        ious = self._iou(_predict[:, 0:4], _gt[:, 0:4])
        for i in range(_predict.shape[0]):
            gt_ious = ious[i, :]
            max_iou, max_index = gt_ious.max(), gt_ious.argmax()
            if max_iou >= iou_thresh:
                if keep[max_index] == 0:
                    image_gt_right[max_index] = -1
                    proposal[i] = -1
                elif image_gt_right[max_index] == 0:
                    image_gt_right[max_index] = 1

            right_index = np.where(image_gt_right == 1)[0]
            image_p_right[i] = len(right_index)



        image_pr = np.zeros((section_num, 2), dtype=np.float)
        for section in range(section_num):
            _thresh = 1 - (section + 1)/section_num
            over_score_index = np.where(predict[:, 4] >= _thresh)[0]
            if over_score_index.shape[0] <= 0:
                image_pr[section, 0] = 0
                image_pr[section, 1] = 0
            else:
                index = over_score_index[-1]
                p_num = len(np.where(proposal[0:(index+1)] == 1)[0])
                image_pr[section, 0] = p_num
                image_pr[section, 1] = image_p_right[index]

        return image_pr


    def get_eval_result(self):
        self._norm_pre_score()
        facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list = self._get_gt_boxes()
        section_num = 1000
        sets = ['easy', 'medium', 'hard']
        set_gts = [easy_gt_list, medium_gt_list, hard_gt_list]
        ap_key_dict = {0: "Easy   Val AP : ", 1: "Medium Val AP : ", 2: "Hard   Val AP : ",}
        ap_dict = {}
        for _set in range(len(sets)):
            gt_list = set_gts[_set]
            count_gt = 0
            pr_curve = np.zeros((section_num, 2), dtype=np.float)
            for i, _ in enumerate(event_list):
                event = str(event_list[i][0][0])
                image_list = file_list[i][0]
                event_predict_dict = self.results[event]
                event_gt_index_list = gt_list[i][0]
                event_gt_box_list = facebox_list[i][0]

                for j, _ in enumerate(image_list):
                    predict = np.array(event_predict_dict[str(image_list[j][0][0])]['bboxes']).astype(np.float)
                    gt_boxes = event_gt_box_list[j][0].astype('float')
                    keep_index = event_gt_index_list[j][0]
                    count_gt += len(keep_index)

                    if gt_boxes.shape[0] <= 0 or predict.shape[0] <= 0:
                        continue
                    keep = np.zeros(gt_boxes.shape[0])
                    if keep_index.shape[0] > 0:
                        keep[keep_index-1] = 1

                    image_pr = self._image_eval(predict, gt_boxes, keep,
                                                iou_thresh=self.iou_thresh,
                                                section_num=section_num)
                    pr_curve += image_pr

            precision = pr_curve[:, 1] / pr_curve[:, 0]
            recall = pr_curve[:, 1] / count_gt

            precision = np.concatenate((np.array([0.]), precision, np.array([0.])))
            recall = np.concatenate((np.array([0.]), recall, np.array([1.])))
            for i in range(precision.shape[0]-1, 0, -1):
                precision[i-1] = np.maximum(precision[i-1], precision[i])
            index = np.where(recall[1:] != recall[:-1])[0]
            ap = np.sum((recall[index + 1] - recall[index]) * precision[index + 1])


            print(ap_key_dict[_set] + '{:.4f}'.format(ap))

        return ap_dict


def decode_bbox(bbox, priors, var):
    boxes = np.concatenate((
        priors[:, 0:2] + bbox[:, 0:2] * var[0] * priors[:, 2:4],
        priors[:, 2:4] * np.exp(bbox[:, 2:4] * var[1])), axis=1)  # (xc, yc, w, h)
    boxes[:, :2] -= boxes[:, 2:] / 2    # (x0, y0, w, h)
    boxes[:, 2:] += boxes[:, :2]        # (x0, y0, x1, y1)
    return boxes

def decode_landm(landm, priors, var):

    return np.concatenate((priors[:, 0:2] + landm[:, 0:2] * var[0] * priors[:, 2:4],
                           priors[:, 0:2] + landm[:, 2:4] * var[0] * priors[:, 2:4],
                           priors[:, 0:2] + landm[:, 4:6] * var[0] * priors[:, 2:4],
                           priors[:, 0:2] + landm[:, 6:8] * var[0] * priors[:, 2:4],
                           priors[:, 0:2] + landm[:, 8:10] * var[0] * priors[:, 2:4],
                           ), axis=1)

def prior_box(image_sizes, min_sizes, steps, clip=False):
    """prior box"""
    feature_maps = [
        [math.ceil(image_sizes[0] / step), math.ceil(image_sizes[1] / step)]
        for step in steps]

    anchors = []
    for k, f in enumerate(feature_maps):
        for i, j in product(range(f[0]), range(f[1])):
            for min_size in min_sizes[k]:
                s_kx = min_size / image_sizes[1]
                s_ky = min_size / image_sizes[0]
                cx = (j + 0.5) * steps[k] / image_sizes[1]
                cy = (i + 0.5) * steps[k] / image_sizes[0]
                anchors += [cx, cy, s_kx, s_ky]

    output = np.asarray(anchors).reshape([-1, 4]).astype(np.float32)

    if clip:
        output = np.clip(output, 0, 1)

    return output

def non_max_suppression(boxes, conf_thres=0.5, nms_thres=0.3):
    detection = boxes
    # 1、找出该图片中得分大于门限函数的框。在进行重合框筛选前就进行得分的筛选可以大幅度减少框的数量。
    mask = detection[:, 4] >= conf_thres
    detection = detection[mask]
    if not np.shape(detection)[0]:
        return []

    best_box = []
    scores = detection[:, 4]
    # 2、根据得分对框进行从大到小排序。
    arg_sort = np.argsort(scores)[::-1]
    detection = detection[arg_sort]

    while np.shape(detection)[0]>0:
        # 3、每次取出得分最大的框，计算其与其它所有预测框的重合程度，重合程度过大的则剔除。
        best_box.append(detection[0])
        if len(detection) == 1:
            break
        ious = iou(best_box[-1], detection[1:])
        detection = detection[1:][ious<nms_thres]

    return np.array(best_box)


def iou(b1, b2):
    b1_x1, b1_y1, b1_x2, b1_y2 = b1[0], b1[1], b1[2], b1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = b2[:, 0], b2[:, 1], b2[:, 2], b2[:, 3]

    inter_rect_x1 = np.maximum(b1_x1, b2_x1)
    inter_rect_y1 = np.maximum(b1_y1, b2_y1)
    inter_rect_x2 = np.minimum(b1_x2, b2_x2)
    inter_rect_y2 = np.minimum(b1_y2, b2_y2)

    inter_area = np.maximum(inter_rect_x2 - inter_rect_x1, 0) * \
                 np.maximum(inter_rect_y2 - inter_rect_y1, 0)

    area_b1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area_b2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    iou = inter_area / np.maximum((area_b1 + area_b2 - inter_area), 1e-6)
    return iou


def val(image_path):
    ms.set_context(mode=ms.GRAPH_MODE, device_target='CPU', save_graphs=False)

    cfg = cfg_res50

    backbone = resnet50(1001)
    network = RetinaFace(phase='predict', backbone=backbone)
    backbone.set_train(False)
    network.set_train(False)

    # load checkpoint
    assert cfg['val_model'] is not None, 'val_model is None.'
    param_dict = ms.load_checkpoint(cfg['val_model'])   # todo 权重文件路径在config.py里修改
    print('Load trained model done. {}'.format(cfg['val_model']))
    network.init_parameters_data()
    ms.load_param_into_net(network, param_dict)


    #     image_path = os.path.join(testset_folder, 'images', img_name)
    image = cv2.imread(image_path)
    # image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # img = np.float32(img_raw)
    old_image = image.copy()
    image = np.array(image, np.float32)
    # ---------------------------------------------------#
    #   计算scale，用于将获得的预测框转换成原图的高宽
    # ---------------------------------------------------#
    scale = [np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0]]
    scale_for_landmarks = [np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
                           np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
                           np.shape(image)[1], np.shape(image)[0]]

    im_height, im_width, _ = np.shape(image)
    priors = prior_box(image_sizes=(im_height, im_width),
                   min_sizes=[[16, 32], [64, 128], [256, 512]],
                   steps=[8, 16, 32],
                   clip=False)


    # scale = np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]], dtype=img.dtype)
    image -= np.array((104, 117, 123), np.float32)
    # img -= (104, 117, 123)
    image = image.transpose(2, 0, 1)
    image = np.expand_dims(image, 0)
    # timers['forward_time'].start()
    image = Tensor(image)  # [1, c, h, w]
    loc, conf, landms = network(image)  # forward pass
    # timers['forward_time'].end()
    # timers['misc'].start()

    boxes = decode_bbox(np.squeeze(loc.asnumpy(), 0), priors, [0.1, 0.2])
    # boxes = boxes.cpu().numpy()
    conf = np.squeeze(conf.asnumpy(), 0)[:, 1:2]
    landms = decode_landm(np.squeeze(landms.asnumpy(), 0), priors, [0.1, 0.2])
    # landms = landms.cpu().numpy()


    boxes_conf_landms = np.concatenate([boxes, conf, landms], -1)

    boxes_conf_landms = non_max_suppression(boxes_conf_landms)
    if len(boxes_conf_landms) <= 0:
        return old_image
    boxes_conf_landms[:, :4] = boxes_conf_landms[:, :4] * scale
    boxes_conf_landms[:, 5:] = boxes_conf_landms[:, 5:] * scale_for_landmarks

    c = np.where(boxes_conf_landms == np.max(boxes_conf_landms[:, 4]))
    b = np.squeeze(boxes_conf_landms[c[0], :])
    # for b in boxes_conf_landms:
    #     text = "{:.4f}".format(b[4])
    #     b = list(map(int, b))
    #
    #     # b[0]-b[3]为人脸框的坐标，b[4]为得分
    #     cv2.rectangle(old_image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
    #     cx = b[0]
    #     cy = b[1] + 12
    #     cv2.putText(old_image, text, (cx, cy),
    #                 cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
    #
    #     cv2.circle(old_image, (b[5], b[6]), 1, (0, 0, 255), 4)
    #     cv2.circle(old_image, (b[7], b[8]), 1, (0, 255, 255), 4)
    #     cv2.circle(old_image, (b[9], b[10]), 1, (255, 0, 255), 4)
    #     cv2.circle(old_image, (b[11], b[12]), 1, (0, 255, 0), 4)
    #     cv2.circle(old_image, (b[13], b[14]), 1, (255, 0, 0), 4)
    b = list(map(int, b))
    old_image = cv2.cvtColor(old_image, cv2.COLOR_RGB2BGR)
    faceimg = old_image[b[1]:b[3], b[0]:b[2]]
    return  faceimg, old_image


def tensor2image(tensor):
    img = tensor.asnumpy()
    img *= 255
    img.clip(0, 255)
    img = img.astype(np.uint8)
    img = img.transpose((1, 2, 0))
    return img


def test_image(config, img_path):
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    context.set_context(save_graphs=False)

    DCE_net = models.zero_dce()
    DCE_net.set_grad(False)

    param_dict = load_checkpoint(config.pretrain_model)
    load_param_into_net(DCE_net, param_dict)
    data_lowlight = Image.open(img_path)
    # data_lowlight = data_lowlight.resize((224, 224), Image.ANTIALIAS)
    data_lowlight = (np.asarray(data_lowlight) / 255.0)
    data_lowlight = data_lowlight.transpose(2, 0, 1)
    data_lowlight = np.array(data_lowlight, np.float32)
    data_lowlight = Tensor(np.expand_dims(data_lowlight, 0))
    y, _ = DCE_net(data_lowlight)

    # a = np.squeeze(y).transpose(1, 2, 0).asnumpy()
    # plt.imshow(a.asnumpy())
    # plt.show()

    b = np.squeeze(y)
    img = tensor2image(b)
    img = Image.fromarray(img.astype('uint8')).convert('RGB')
    return img



if __name__ == '__main__':

# def fatigue_detection(img):

    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--lowlight_images_path', type=str, default="data/test_data/real/", help="the test data path.")
    parser.add_argument('--batch_size', type=int, default=8, help="batch size")
    parser.add_argument('--pretrain_model', type=str, default='./pretrain_model/zero_dce_epoch99.ckpt',
                        help="the pretrain model path")
    parser.add_argument('--save_with_src', action='store_true', help="whether save the source image")
    parser.add_argument('--save_path', type=str, default='./outputs/real', help="the output dir")
    parser.add_argument('--image_type', type=str, default="png", help="the image postfix")

    configdg = parser.parse_args()

    if not os.path.exists(configdg.save_path):
        os.makedirs(configdg.save_path)

    plt.subplot(2, 2, 1)
    plt.title("source image")
    data_lowlight = Image.open('test.jpg')
    # data_lowlight = img
    plt.imshow(data_lowlight)


    '''图像增强'''
    img = test_image(configdg, 'test.jpg')
    img.save('image/04.jpg')
    plt.subplot(2, 2, 2)
    plt.title("enhance image")
    plt.imshow(img)

    '''人脸识别'''
    face_img, image = val('image/04.jpg') # todo 输入图片位置（如 ‘img\face.jpg')
    plt.subplot(2, 2, 3)
    plt.title("face identification")
    plt.imshow(face_img)

    h = face_img.shape[0]
    w = face_img.shape[1]
    y0 = int(h / 3 - h / 15)
    y1 = int(h / 3 + h / 5)
    x0 = int(w / 5 - w / 5)
    x1 = int(w / 5 + w / 3)

    eye_img = face_img[y0:y1, x0:x1]

    # cv2.imshow("cut_image", eye_img)
    # cv2.imshow("result", face_img)

    # cv2.imshow("result", image)
    # 保存图片
    # w_image = cv2.imwrite('face/01.jpg', eye_img)
    # cv2.waitKey(0)

    eye = Image.fromarray(eye_img.astype('uint8')).convert('RGB')
    eye.save('eye/01.jpg')
    image = Image.open('eye/01.jpg')
    plt.subplot(2, 2, 4)
    plt.title("eye recognition")
    plt.imshow(image)
    image = ds.vision.c_transforms.Resize((256, 256))(image)
    image = ds.vision.c_transforms.CenterCrop(224)(image)
    image = ds.vision.c_transforms.Normalize(mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                             std=[0.229 * 255, 0.224 * 255, 0.225 * 255])(image)
    image = ds.vision.c_transforms.HWC2CHW()(image)
    image = np.array(image, np.float32)
    image = np.expand_dims(image, 0)
    image = Tensor(image)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    _, _, net = define_net(config, is_training=True)
    load_ckpt(net, config.pretrain_ckpt)
    net.set_train(False)

    res = np.squeeze(net(image))
    predict = ops.Softmax(axis=0)(res)
    predict_cla = ops.Argmax(axis=0)(Tensor(predict))
    class_indict = class_indict[str(predict_cla)]
    predict = predict[predict_cla].asnumpy()

    print_res = "class: {}   prob: {:.5}".format(class_indict, predict)
    plt.title(print_res)
    print(print_res)
    plt.show()

    # return class_indict, predict
