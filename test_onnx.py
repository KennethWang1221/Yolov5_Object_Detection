#!/usr/bin/env python3

import sys
import argparse
import cv2
import os
import numpy as np
import glob
import onnxruntime
import matplotlib.pyplot as plt


def check_tensor(tensor):
    if isinstance(tensor,list):
        print("Type:",type(tensor))
        print("len:",len(tensor))
    else:
        print("Type:",type(tensor))
        print("Shape:",tensor.shape)
        print("dtype:",tensor.dtype)

def diff_tensor(img1,img2):
    diff = img1-img2
    print("max:",np.max(diff))
    print("minx:",np.min(diff))
    print("mean:",np.mean(abs(diff)))
    print((img1==img2).all())

def show_img(img):
    plt.figure()
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
def normalization(args, image):
    mean =np.array(args['mean'])
    scale =np.array(args['scale'])
    image = np.float32(image)

    if args['file_format'] == 'nchw':
        image = image - mean[:,np.newaxis,np.newaxis]
        image = image * scale[:,np.newaxis,np.newaxis]    
    elif args['file_format'] == 'nhwc':
        image = image - mean[np.newaxis,np.newaxis,:]
        image = image * scale[np.newaxis,np.newaxis,:]

    return np.float32(image)

def set_image_format(image,file_format,channel_order):
    if channel_order == 'rgb':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channel = image.shape
        if file_format == 'nhwc':
            pass
        else:
            image = np.transpose(image,(2,0,1))
    elif channel_order == 'bgr':
        height, width,channel = image.shape
        if file_format == 'nhwc':
            pass
        else:
            image = np.transpose(image,(2,0,1))
    elif channel_order == 'gray':
        height, width, channel = image.shape
        image = image[:,:,0]
    else:
        print('{} file type is not supported'.format(image))

    return image


def colors(i, bgr=False):
    hex_colors = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                  '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
    palette = [tuple(int(h[i:i+2], 16) for i in (0, 2, 4)) for h in hex_colors]
    n = len(palette)
    c = palette[int(i) % n]
    return (c[2], c[1], c[0]) if bgr else c

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

def nms(boxes, scores, iou_threshold):
    x1 = boxes[:, 0] # e.g, (40,)
    y1 = boxes[:, 1] # e.g, (40,)
    x2 = boxes[:, 2] # e.g, (40,)
    y2 = boxes[:, 3] # e.g, (40,)
    areas = (x2 - x1 + 1) * (y2 - y1 + 1) # if boxes: [40,4], areas will be: [40,]

    scores_order = scores.argsort()[::-1] # (40,) This gives us the indices of the boxes sorted by score in descending order

    keep = []
    while scores_order.size > 0:
        i = scores_order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[scores_order[1:]]) # (39,) xx1 is computed as the maximum of the x-coordinate of the box with the highest score and the x-coordinates of the remaining boxes.
        yy1 = np.maximum(y1[i], y1[scores_order[1:]]) # (39,) yy1 is computed as the maximum of the y-coordinate of the box with the highest score and the y-coordinates of the remaining boxes. The reason for using np.maximum for xx1 and yy1 is to compute the coordinate values for the top-left corner of the intersection boxes. Since the iou calculation involves computing the intersection between the current box and other boxes, we need to compute the top-left coordinates of the intersection boxes as the maximum of the top-left coordinates of the two intersecting boxes. On the other hand, the reason for using np.minimum for xx2 and yy2 is to compute the coordinate values for the bottom-right corner of the intersection boxes. Since the iou calculation involves computing the intersection between the current box and other boxes, we need to compute the bottom-right coordinates of the intersection boxes as the minimum of the bottom-right coordinates of the two intersecting boxes. By doing so, we ensure that the intersection boxes are properly defined and do not contain any areas outside the two intersecting boxes.
        xx2 = np.minimum(x2[i], x2[scores_order[1:]]) # (39,) xx2 is computed as the minimum of the x-coordinate of the box with the highest score and the x-coordinates of the remaining boxes. 
        yy2 = np.minimum(y2[i], y2[scores_order[1:]]) # (39,) yy2 is computed as the minimum of the y-coordinate of the box with the highest score and the y-coordinates of the remaining boxes.

        w = np.maximum(0.0, xx2 - xx1 + 1) # (39,)
        h = np.maximum(0.0, yy2 - yy1 + 1) # (39,)
        inter = w * h # (39,)

        iou = inter / (areas[i] + areas[scores_order[1:]] - inter) # (39,) areas[i] represents the area of the i-th box. areas[scores_order[1:]] represents the areas of the rest of the boxes in boxes that have not been selected as a part of the NMS. inter represents the area of the intersection between the i-th box and the rest of the boxes in boxes that have not been selected as a part of the NMS. Therefore, areas[i] + areas[scores_order[1:]] - inter calculates the total area covered by the i-th box and the rest of the boxes that have not been selected as a part of the NMS. 
        
        temp = np.where(iou <= iou_threshold)[0] + 1 # (25,) np.where return indices
        scores_order = scores_order[temp] # (25,) In this line, we select the indices of bounding boxes whose IoU with the current bounding box i is less than or equal to the iou_threshold. np.where(iou <= iou_threshold)[0] returns an array of indices where the condition iou <= iou_threshold is true. Then we add 1 to all these indices using + 1. This is because the first element of scores_order is already appended to keep, so we need to start iterating from the next element. Finally, we use this updated scores_order array to select the next highest score box for comparison with the current box. The condition iou <= iou_threshold is used to filter out all boxes with IOU (Intersection over Union) overlap greater than a certain threshold iou_threshold. When performing NMS, the algorithm looks at the boxes with the highest scores first and checks if there is significant overlap between the current box and the other boxes with higher scores. If there is significant overlap (i.e., IOU is greater than the threshold), then the algorithm removes the overlapping boxes from consideration and moves on to the next highest-scoring box. By doing so, the algorithm ensures that only the most confident and diverse set of boxes are selected, and redundant boxes are eliminated. In short, iou <= iou_threshold is used to identify the boxes that significantly overlap with the current box and eliminate them from consideration during NMS.

    return np.array(keep)


def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width] [576, 768]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape) # (640, 640)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1]) # 1.1111111111111112 0.8333333333333334
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios # (0.8333333333333334, 0.8333333333333334)
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r)) # 640 480
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding dw:0 dh: 160 new_shape[1] 640 new_shape[0] 640 new_unpad[0] 640 new_unpad[1] 480
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize  shape (576, 768) shape[::-1] (768, 576) ew_unpad (640, 480)
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR) # im: (480, 640, 3)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border im shape: 640 640 3 uint8
    return im, ratio, (dw, dh) # (0.8333333333333334, 0.8333333333333334) (0.0, 80.0)

def scale_coords(args, img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if args['use_letterbox']:
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        coords[:, [0, 2]] -= pad[0]  # x padding
        coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, :4] /= gain

    if args['resize']:
        y_ratio, x_ratio = (img1_shape[0] / img0_shape[0]), (img1_shape[1] / img0_shape[1])  # gain  = old / new
        coords[:, [0, 2]] /= x_ratio  # x padding
        coords[:, [1, 3]] /= y_ratio  # y padding

    clip_coords(coords, img0_shape)

    return coords

def box_label(im, line_width, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
    
    lw = line_width or max(round(sum(im.shape) / 2 * 0.003), 2)
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(im, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    if label:
        tf = max(lw - 1, 1)  # font thickness
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
        outside = p1[1] - h - 3 >= 0  # label fits outside box
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(im, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im,
                    label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    0,
                    lw / 3,
                    txt_color,
                    thickness=tf,
                    lineType=cv2.LINE_AA)

    return np.asarray(im)

def non_max_suppression(prediction, 
                        max_wh,
                        max_nms,
                        redundant,
                        merge,
                        conf_thres=0.25,
                        iou_thres=0.45,
                        classes=None,
                        agnostic=False,
                        multi_label=False,
                        labels=(),
                        max_det=300):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping bounding boxes

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    # prediction: [1,25200, 85] (num_boxes, 85), where num_boxes is the number of predicted boxes per image and 85 is the total number of predicted values for each box (4 for the box coordinates, 1 for the objectness score, and 80 for the class probabilities).
    bs = prediction.shape[0]  # 1, batch size bs is set to the first dimension of the prediction tensor, representing the number of images in the batch.
    nc = prediction.shape[2] - 5  # 80, number of classes nc is set to the number of classes predicted by the model. The value of 5 is subtracted because the first 5 elements of each bounding box prediction correspond to the coordinates and objectness score and not the classes.
    xc = prediction[..., 4] > conf_thres  # (1, 25200) bool  candidates xc is set to a boolean tensor that represents the candidates with a confidence score greater than the conf_thres input.

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    # time_limit = 0.1 + 0.03 * bs  # seconds to quit after
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    output = [np.zeros((0, 6))] * bs # [x1, y1, x2, y2, conf, class] If no detections are made for an image, its corresponding array will remain empty with shape (0,6).
    for xi, x in enumerate(prediction):  # image index, image inference (xi==0, x== (25200, 85))
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence ((25200, 85)->select-> (45, 85)), xc[xi]: (1,25200) -> (25200,) bool type, By indexing x with xc[xi], this line keeps only the bounding boxes with high confidence scores (confidence score greater than the conf_thres input).

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = np.zeros((len(lb), nc + 5))
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].astype(np.int64) + 5] = 1.0  # cls
            x = np.concatenate((x, v), axis=0)
        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # (45, 80) * (45, 1) == (45,80) conf = obj_conf * cls_conf (num_boxes, 1) * (num_boxes, 80). The expression x[:, 5:] selects all columns of x starting from index 5 (i.e., the class probability scores), while x[:, 4:5] selects only the objectness score (which is in column 4). By multiplying these two arrays element-wise (*=), we are effectively weighting the class probabilities by the objectness score. This step is necessary to ensure that the class probabilities are weighted by the objectness score. The idea is that if the objectness score of a bounding box is low, it is less likely to contain an object of interest. Therefore, the class probabilities for that bounding box should be down-weighted to reflect this uncertainty. Conversely, if the objectness score is high, the class probabilities should be up-weighted to reflect the confidence that the bounding box contains an object of interest.
        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4]) # (45,4) -> (45,4)

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = np.where(x[:, 5:] > conf_thres)
            x = np.concatenate((box[i], x[i, j + 5][:, None], j[:, None].astype(np.float32)), axis=1)
        else:  # best class only
            conf = x[:, 5:].max(axis=1) # (45,80)->(45,) finds the maximum confidence score for each bounding box
            j = np.argmax(x[:, 5:], axis=1) # (45,80)->(45,) finds the index of the class with the maximum confidence score for each bounding box.
            conf = np.expand_dims(conf, axis=1) # (45,1)
            j = np.expand_dims(j, axis=1) # (45,1)
            # e.g., x: [45,85] -> [45,6] -> [40,6]
            x = np.concatenate((box, conf, j.astype(np.float32)), 1) # (45,6)
            x = x[conf.ravel() > conf_thres] # (40, 6) 1. np.concatenate((box, conf, j.astype(np.float32)), 1) concatenates the boxes, their confidence scores, and their class indices along the second axis (i.e., columns). 2. [conf.ravel() > conf_thres] creates a boolean mask that indicates which boxes have a confidence score greater than the specified threshold conf_thres. 3. x[conf.ravel() > conf_thres] applies the boolean mask to the concatenated array to select only those boxes with high enough confidence scores. Finally, the resulting selected boxes are assigned to the variable 'x'

        # Filter by class
        if classes is not None:
            # for example classes = 1 
            x = x[np.isin(x[:, 5:6], np.array(classes, dtype=x.dtype))[:, 0]]
        else:
            pass

        n = x.shape[0]  # number of boxes e.g., x
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort()[::-1][:max_nms]] 

        c = x[:, 5:6] * (0 if agnostic else max_wh)  # (40,1) classes In the YOLOv5 NMS function, c represents the class-specific confidence scores, which are calculated as the element-wise multiplication of the original confidence scores (x[:, 5:]) with the maximum box width and height (max_wh). The purpose of multiplying the confidence scores by the maximum box width and height is to give higher scores to boxes that are wider and taller, and lower scores to boxes that are smaller in size. This is done to prioritize larger boxes over smaller ones during the NMS process, as larger boxes are generally considered to be more confident detections.
        boxes, scores = x[:, :4] + c, x[:, 4]  # (40,4), (40,1) boxes (offset by class), scores

        i = nms(boxes, scores, iou_thres)  # NMS (5,)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        output[xi] = x[i] # x: (40,6)->(5,6), thus, output[0]: (5,6)
    return output
    
def preprocess(args, file_path):
    ori_file_name_list = []
    preprocessed = []
    ori_image_list = []
    
    file_name =  os.path.splitext(os.path.basename(file_path))[0] 
    ori_file_name_list.append(file_path)
    
    img0 = cv2.imread(file_path)
    ori_image_list.append(img0)
    
    if args['use_letterbox']:
        im = letterbox(img0, args['imgsz'], stride=args['stride'], auto=False)[0]
    else:
        if args['resize']:
            im = cv2.resize(img0, (args['imgsz'][0], args['imgsz'][1])) 
        else:
            im = img0.copy()

    if args['channel_order'] == 'rgb':
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    if args['file_format'] == 'nchw':
        im = np.transpose(im,(2,0,1))
    
    if args['normalize']:
        im = normalization(args, im)

    preprocessed.append(im)
        
    return np.array(preprocessed), np.array(ori_image_list), ori_file_name_list

def yolov5_inference(args, model, preprocessed, ori_file_name_list, inference_input_tensor_path, inference_opt_tensor_path):
    first_input_name = model.get_inputs()[0].name
    first_output_name = model.get_outputs()[0].name
    result_list = []
    for i in range(preprocessed.shape[0]):  
        file_name = os.path.splitext(os.path.basename(ori_file_name_list[i]))[0]
        if len(preprocessed[i].shape) == 3:
            im = np.expand_dims(preprocessed[i], axis=0)  # expand for batch dim

        np.savetxt('{}/{}_input.tensor'.format(inference_input_tensor_path, file_name,),
            preprocessed[i].reshape(-1), fmt='%.6f')

        output = model.run([first_output_name], {
                first_input_name: im})
        np.savetxt('{}/{}_output_{}.tensor'.format(inference_opt_tensor_path, file_name, 0),
                output[0].reshape(-1), fmt='%.6f')

        result_list.append(output[0])

    data = result_list
    return data
    
def build_onnx_model(onnx_file):
    session = onnxruntime.InferenceSession(onnx_file)
    session.get_modelmeta()
    first_input_name = session.get_inputs()[0].name
    first_output_name = session.get_outputs()[0].name

    meta = session.get_modelmeta().custom_metadata_map  # metadata
    if 'stride' in meta:
        stride, names = int(meta['stride']), eval(meta['names'])

    return session
    
def get_anchors(anchors_arr,grid_h):

    anchors_arr_0 = np.zeros((grid_h,grid_h),dtype=int) # (shape, shape)
    anchors_arr_0 = np.expand_dims(anchors_arr_0,axis=2) # (shape, shape, 1)
    anchors_arr_0 = np.concatenate((anchors_arr_0, anchors_arr_0), axis=-1) # (shape, shape, 2)
    for i in range(grid_h):
        anchors_arr_0[i,:] = anchors_arr[0] # anchors_arr_0: (1,20,20,2) anchors_arr[i,:,:] (shape,2) = (2,)
    anchors_arr_0 = np.expand_dims(anchors_arr_0,axis=0) # (1, shape, shape, 2)


    anchors_arr_1 = np.zeros((grid_h,grid_h),dtype=int) # (shape, shape)
    anchors_arr_1 = np.expand_dims(anchors_arr_1,axis=2) # (shape, shape, 1)
    anchors_arr_1 = np.concatenate((anchors_arr_1, anchors_arr_1), axis=-1) # (shape, shape, 2)
    for i in range(grid_h):
        anchors_arr_1[i,:] = anchors_arr[1]
    anchors_arr_1 = np.expand_dims(anchors_arr_1,axis=0) # (1, shape, shape, 2)


    anchors_arr_2 = np.zeros((grid_h,grid_h),dtype=int)
    anchors_arr_2 = np.expand_dims(anchors_arr_2,axis=2)
    anchors_arr_2 = np.concatenate((anchors_arr_2, anchors_arr_2), axis=-1) # (1, shape, shape, 2)
    for i in range(grid_h):
        anchors_arr_2[i,:] = anchors_arr[2]
    anchors_arr_2 = np.expand_dims(anchors_arr_2,axis=0)

    anchors_arr_final = np.concatenate((anchors_arr_0, anchors_arr_1, anchors_arr_2), axis=0) # (3, shape, shape, 2)
    anchors_arr_final = np.expand_dims(anchors_arr_final,axis=0) # (1, 3, 80, 80, 2) # (1, 3, shape, shape, 2)
    
    return anchors_arr_final
    
def yolov5_concat_tail(input_data,masks, anchors_list):
    output_20_40_80 = []
    for input,mask in zip(input_data, masks): # (1,3,20,20,85) (6,7,8) / (1,3,40,40,85) (3,4,5) / (1,3,80,80,85) (1,2,3)

        box_xy = sigmoid(input[..., :2]) # (1,3,shape,shape,2) shape==20/40/80
        box_wh = sigmoid(input[..., 2:4]) # (1,3,shape,shape,2)
        box_confidence = sigmoid(input[..., 4:]) # (1,3,shape,shape,81)

        # grid for box_xy 
        grid_h = int(input.shape[2]) # (shape,shape)
        grid_w = int(input.shape[2]) # (shape,shape)
        col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w) # (shape*shape) -> (shape, shape)
        row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h) # (shape*shape) -> (shape, shape)
        col = col.reshape(1, grid_h, grid_w, 1).repeat(3, axis=-4) # (1, shape, shape, 1) -> (3, shape, shape, 1)
        row = row.reshape(1, grid_h, grid_w, 1).repeat(3, axis=-4) # (1, shape, shape, 1) -> (3, shape, shape, 1)
        grid = np.concatenate((col, row), axis=-1) # (3, shape, shape, 2)
        grid = np.expand_dims(grid,axis=0) # (1, 3, shape, shape, 2)
        grid = grid - 0.5

        box_xy = box_xy * 2
        box_xy = box_xy + grid
        
        if grid_w == 80:
            box_xy = box_xy * 8
        if grid_w == 40:
            box_xy = box_xy * 16
        if grid_w == 20:
            box_xy = box_xy * 32 # (1, 3, shape, shape, 2)

        # anchors for box_wh
        anchors = [anchors_list[i] for i in mask]
        anchors_arr = np.array(anchors) # (3,2)
        anchors_arr_final = get_anchors(anchors_arr,grid_h) # (1,3,shape,shape,2)
        box_wh = box_wh * 2
        box_wh = pow(box_wh,2) # (1,3,shape,shape,2)
        box_wh = box_wh * anchors_arr_final

        # concatenate box_xy, box_wh, box_confidence
        output = np.concatenate((box_xy, box_wh, box_confidence), axis=-1) # (1,3,shape,shape,85)
        output = output.reshape(-1,(output.shape[1]*output.shape[2]*output.shape[3]),85) # (1,3,shape1*shape2*shape3,85)
        
        # append (1, 3*80*80, 85), (1, 3*40*40, 85),  (1, 3*20*20, 85) to output_20_40_80 list
        output_20_40_80.append(output)
        
    # concatenate  (1, 19200, 85),(1, 4800, 85), (1, 1200, 85) to output_con (1,25200,85)
    output_con = np.concatenate((output_20_40_80[2],output_20_40_80[1], output_20_40_80[0]),axis = -2)
    return output_con


def yolov5_concat(args, inference_res, ori_file_name_list, inference_opt_tensor_path):
    shape = len(args['classes_names']) + 5
    for i in range(len(inference_res)):
        file_name = os.path.splitext(os.path.basename(ori_file_name_list[i]))[0]
        tensor0 = inference_res[i][1].reshape(-1, 3, shape, 20, 20) 
        tensor0 = np.transpose(tensor0, (0, 1, 3, 4, 2)) # (1, 3, 20, 20, 85)
        tensor0 = tensor0.reshape(-1,3,20,20, shape) # (1, 3, 20, 20, 85)

        tensor1 = inference_res[i][0].reshape(-1, 3, shape, 40, 40) 
        tensor1 = np.transpose(tensor1, (0, 1, 3, 4, 2)) # (1, 3, 40, 40, 85)
        tensor1 = tensor1.reshape(-1,3,40,40,shape) # (1, 3, 40, 40, 85)

        tensor2 = inference_res[i][2].reshape(-1, 3, shape, 80, 80) 
        tensor2 = np.transpose(tensor2, (0, 1, 3, 4, 2)) # (1, 3, 80, 80, 85)
        tensor2 = tensor2.reshape(-1,3,80,80,shape) # (1, 3, 80, 80, 85)
    
        input_data = []
        input_data.append(tensor0)
        input_data.append(tensor1)
        input_data.append(tensor2)
        output_con = yolov5_concat_tail(input_data,args['masks'],args['anchors_list'])

        if args['save_concat']:
            save_path = os.path.join(inference_opt_tensor_path, '{}_concat.{}'.format(file_name,'tensor'))
            np.savetxt(save_path, output_con.reshape(-1,1))
            # output_con_tensor = np.loadtxt(save_path)
            # output_con_tensor_reshape = output_con_tensor.reshape(1,25200,7)
    return output_con

def yolov5_postprocess(args, output_con, preprocessed, ori_image_list, ori_file_name_list, image_results_path, txt_results_path):
    pred = non_max_suppression(output_con,args['max_wh'], args['max_nms'], args['redundant'], args['merge'], args['conf_thres'], args['iou_thres'], args['classes'], args['agnostic_nms'], args['max_det'])
    
    for i in range(len(ori_image_list)):

        file_name = os.path.splitext(os.path.basename(ori_file_name_list[i]))[0]
        image_path = os.path.join(image_results_path, '{}.png'.format(file_name)) 
        txt_path = os.path.join(txt_results_path, '{}.txt'.format(file_name)) 
        im0s = ori_image_list[i]
        im = preprocessed[0]
        for i, det in enumerate(pred):
            im0 = im0s.copy()
            gn = np.array([im0.shape[1], im0.shape[0], im0.shape[1], im0.shape[0]]) # [768 576 768 576]
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(args, im.shape[1:], det[:, :4], im0.shape).round() # (640, 640), (5, 4), (576, 768, 3) -> (5,4)

                for c in np.unique(det[:, -1]): # c == [1,2,7,16,58] The np.unique() function returns an array of unique values in the input array in sorted order.
                    n = np.sum(det[:, -1] == c)  # detections per class n == 1

                # Write results
                classes_names = args['classes_names']
                with open(txt_path, 'a') as f:
                    for *xyxy, conf, cls in reversed(det): # (5,6) # *xyxy: <class 'list'> len:4 [56.65332, 81.46141, 110.08868, 135.83157]/ conf: <class 'numpy.float32'> 0.2649595 / cls: <class 'numpy.float32'> 58.0         
                        c = int(cls)  # integer class
                        xywh = (xyxy2xywh(np.array(xyxy).reshape(1, 4)) / gn).reshape(-1).tolist()  # normalized xywh
                        line = (cls, *xyxy, conf)  # label format
                        temp_info = ('%g ' * len(line)).rstrip() % line + '\n'
                        temp_info = classes_names[c] + ":" + temp_info
                        f.write(temp_info)
                        
                        label = (f'{classes_names[c]} {conf:.2f}')
                        im0 = box_label(im0, args['line_thickness'], xyxy, label, color=colors(c, True), txt_color=(255, 255, 255))
            
            cv2.imwrite(image_path, im0)

def main(**args):
    if not os.path.isdir(args['test_path']):
        raise ValueError("Invalid test intput dir `"+os.path.abspath(args['test_path'])+"`")

    if os.access(args['onnx_file'], os.R_OK) == 0:
        print('cannot access network binary {}'.format(args['model_file']))
        sys.exit(1)

    if not os.path.exists(args['opts_dir']):
        os.makedirs(args['opts_dir'])

    inference_opt_tensor_path = os.path.join(args['opts_dir'], 'inference_opt_tensor')
    image_results_path = os.path.join(args['opts_dir'], 'image_res') 
    txt_results_path = os.path.join(args['opts_dir'], 'txt_res') 
    inference_input_tensor_path = os.path.join(args['opts_dir'], 'inference_input_tensor')

    if not os.path.exists(inference_opt_tensor_path):
        os.makedirs(inference_opt_tensor_path)
    if not os.path.exists(image_results_path):
        os.makedirs(image_results_path)
    if not os.path.exists(txt_results_path):
        os.makedirs(txt_results_path)
    if not os.path.exists(inference_input_tensor_path):
        os.makedirs(inference_input_tensor_path)
    
    filelist = sorted(glob.glob(args['test_path'] + '/*[.png, .jpg]'))
    model = build_onnx_model(args['onnx_file'])

    for index, file in enumerate(filelist):
    
        preprocessed, ori_image_list, ori_file_name_list = preprocess(args, file)

        inference_res = yolov5_inference(args, model, preprocessed, ori_file_name_list, inference_input_tensor_path, inference_opt_tensor_path)
        
        #output_con = yolov5_concat(args, inference_res, ori_file_name_list, inference_opt_tensor_path)
        
        yolov5_postprocess(args, inference_res[0], preprocessed, ori_image_list, ori_file_name_list, image_results_path, txt_results_path)


if  __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AI Python Inference API for yolov5 demo')
    # Load file
    parser.add_argument("--onnx_file", type=str,default="./yolov5_onnx.onnx", \
                        help='path to NBG model')
    parser.add_argument("--test_path", type=str, default="./data/", \
                        help='path to image folder')
    parser.add_argument("--conf_thres", type=float, default=0.25, \
                        help='define cutoff prob')
    parser.add_argument("--iou_thres", type=float, default=0.45, \
                        help='display positive')
    parser.add_argument("--mean", nargs='+', type=int,default=[0.0, 0.0, 0.0], 
                        help='value of mean for model')
    parser.add_argument("--scale", nargs='+', type=int,default=[0.00392157], 
                        help='value of scale for model')
    parser.add_argument('--file_format', '-f', type=str, default='nchw',
                        help='specify the model input format')
    parser.add_argument('--channel_order', '-c', type=str, default='rgb',
                        help='specify the order of channels')
    parser.add_argument('--classes', nargs='+', type=int, 
                        help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--use_letterbox', action='store_true', 
                        help='use letterbox')
    parser.add_argument('--resize', action='store_false', 
                        help='resize image to model size')
    parser.add_argument('--normalize', action='store_false', 
                        help='normalize image ')
    parser.add_argument("--max_wh", nargs='+', type=int,default=7680, 
                        help='(pixels) maximum box width and height')
    parser.add_argument("--max_nms", nargs='+', type=int,default=30000, 
                        help='maximum number of boxes into torchvision.ops.nms()')
    parser.add_argument("--stride", nargs='+', type=int,default=32, 
                        help='model stride')
    parser.add_argument("--classes_names", nargs='+', type=float, default=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'], \
                        help='yolov5 claees name')
    parser.add_argument("--imgsz", nargs='+', type=int, default=[640,640], 
                        help='size of target image')
    parser.add_argument("--line_thickness", nargs='+', type=int,default=3, 
                        help='line thickness')
    parser.add_argument("--masks", nargs='+', type=float, default=[[6, 7, 8], [3, 4, 5], [0, 1, 2]], \
                        help='yolov5 mask list')
    parser.add_argument("--anchors_list", nargs='+', type=float, default=[[10, 13], [16, 30], [33,23], [30,61], [62,45], [59,119],[116,90],[156,198],[373,326]], \
                        help='yolov5 anchor list')
    parser.add_argument('--max_det', type=int, default=1000, 
                        help='maximum detections per image')
    parser.add_argument('--agnostic_nms', action='store_true', 
                        help='class-agnostic NMS')
    parser.add_argument('--agnostic', action='store_true', 
                        help='agnostic NMS')
    parser.add_argument('--multi_label', action='store_true', 
                        help='multiple labels per box')
    parser.add_argument('--save_conf', action='store_false', 
                        help='save confidences in --save-txt labels')
    parser.add_argument('--save_crop', action='store_true', 
                        help='save confidences in --save-txt labels')
    parser.add_argument('--save_txt', action='store_false', 
                        help='save confidences in --save-txt labels')
    parser.add_argument('--save_img', action='store_false', 
                        help='save confidences in --save-txt labels')
    parser.add_argument('--hide_labels', action='store_true', 
                        help='hide labels')
    parser.add_argument('--hide_conf', action='store_true', 
                        help='hide confidences')
    parser.add_argument('--redundant', action='store_true', 
                        help='require redundant detections')
    parser.add_argument('--merge', action='store_true', 
                        help='use merge-NMS')
    parser.add_argument('--save_concat',  action='store_true', default=False,\
                        help='save concatenate tensor')
    # Dirs
    parser.add_argument("--opts_dir", type=str, default="./api_res", \
                        help='path of outputs files ')
    argspar = parser.parse_args()    

    print("\n### Test yolov5 NBG model ###")
    print("> Parameters:")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')

    main(**vars(argspar))













