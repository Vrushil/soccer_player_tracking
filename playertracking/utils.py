from absl import logging
import numpy as np
import tensorflow as tf
import cv2
import time
import copy

YOLOV3_LAYER_LIST = [
    'yolo_darknet',
    'yolo_conv_0',
    'yolo_output_0',
    'yolo_conv_1',
    'yolo_output_1',
    'yolo_conv_2',
    'yolo_output_2',
]

YOLOV3_TINY_LAYER_LIST = [
    'yolo_darknet',
    'yolo_conv_0',
    'yolo_output_0',
    'yolo_conv_1',
    'yolo_output_1',
]


def load_darknet_weights(model, weights_file, tiny=False):
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

    if tiny:
        layers = YOLOV3_TINY_LAYER_LIST
    else:
        layers = YOLOV3_LAYER_LIST

    for layer_name in layers:
        sub_model = model.get_layer(layer_name)
        for i, layer in enumerate(sub_model.layers):
            if not layer.name.startswith('conv2d'):
                continue
            batch_norm = None
            if i + 1 < len(sub_model.layers) and \
                    sub_model.layers[i + 1].name.startswith('batch_norm'):
                batch_norm = sub_model.layers[i + 1]

            logging.info("{}/{} {}".format(
                sub_model.name, layer.name, 'bn' if batch_norm else 'bias'))

            filters = layer.filters
            size = layer.kernel_size[0]
            in_dim = layer.input_shape[-1]

            if batch_norm is None:
                conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)
            else:
                # darknet [beta, gamma, mean, variance]
                bn_weights = np.fromfile(
                    wf, dtype=np.float32, count=4 * filters)
                # tf [gamma, beta, mean, variance]
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]

            # darknet shape (out_dim, in_dim, height, width)
            conv_shape = (filters, in_dim, size, size)
            conv_weights = np.fromfile(
                wf, dtype=np.float32, count=np.product(conv_shape))
            # tf shape (height, width, in_dim, out_dim)
            conv_weights = conv_weights.reshape(
                conv_shape).transpose([2, 3, 1, 0])

            if batch_norm is None:
                layer.set_weights([conv_weights, conv_bias])
            else:
                layer.set_weights([conv_weights])
                batch_norm.set_weights(bn_weights)

    assert len(wf.read()) == 0, 'failed to read all data'
    wf.close()


def broadcast_iou(box_1, box_2):
    # box_1: (..., (x1, y1, x2, y2))
    # box_2: (N, (x1, y1, x2, y2))

    # broadcast boxes
    box_1 = tf.expand_dims(box_1, -2)
    box_2 = tf.expand_dims(box_2, 0)
    # new_shape: (..., N, (x1, y1, x2, y2))
    new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
    box_1 = tf.broadcast_to(box_1, new_shape)
    box_2 = tf.broadcast_to(box_2, new_shape)

    int_w = tf.maximum(tf.minimum(box_1[..., 2], box_2[..., 2]) -
                       tf.maximum(box_1[..., 0], box_2[..., 0]), 0)
    int_h = tf.maximum(tf.minimum(box_1[..., 3], box_2[..., 3]) -
                       tf.maximum(box_1[..., 1], box_2[..., 1]), 0)
    int_area = int_w * int_h
    box_1_area = (box_1[..., 2] - box_1[..., 0]) * \
        (box_1[..., 3] - box_1[..., 1])
    box_2_area = (box_2[..., 2] - box_2[..., 0]) * \
        (box_2[..., 3] - box_2[..., 1])
    return int_area / (box_1_area + box_2_area - int_area)


def draw_outputs(img, outputs, class_names):
    boxes, objectness, classes, nums = outputs
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    

    wh = np.flip(img.shape[0:2])
    for i in range(nums):
       
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
            
        if class_names[int(classes[i])]=='person' :
            img = cv2.rectangle(img, x1y1, x2y2, (255, 255, 0), 1)
            #img = cv2.putText(img, '{} {:.2f}'.format(
            #class_names[int(classes[i])], objectness[i]),
            #(x1y1[0],x2y2[1]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0,240 ), 1)
        if class_names[int(classes[i])]=='crowd' :
            img = cv2.rectangle(img, x1y1, x2y2, (0,0 , 255), 1)
            img = cv2.putText(img, '{} {:.2f}'.format(
            class_names[int(classes[i])], objectness[i]),
            (x1y1[0],x2y2[1]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0,0 ), 1)
    return img


def draw_outputs_3(img, outputs, class_names):
    
    #essentials
    boxes, objectness, classes, nums = outputs
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    wh = np.flip(img.shape[0:2])
    print(wh)
    ## customization
    #fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
    fgbg=cv2.createBackgroundSubtractorMOG2()
    last_recorded_time = time.time()
    last_recorded_time_1 = time.time()
    first_iteration_indicator = 1
    for i in range(nums):
       # essentials 
        global crop_img,crop_black
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        
        
        #customization
        curr_time = time.time()
        if first_iteration_indicator == 1:
            first_frame =copy.deepcopy(img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape[:2]
            accum_image = np.zeros((height, width), np.uint8)
            #crop_img=img[x1y1[1]:x2y2[1],x1y1[0]:x2y2[0]]
            #crop_black=accum_image[x1y1[1]:x2y2[1],x1y1[0]:x2y2[0]]
            first_iteration_indicator = 0
        #if curr_time - last_recorded_time_1 >=7.0: 
        #    accum_image = np.zeros((height, width), np.uint8)
        #    print ("Accum Image updated")
        #    last_recorded_time_1 = curr_time
        else:
            #crop_img=img[x1y1[1]:x2y2[1],x1y1[0]:x2y2[0]]
            #crop_black=accum_image[x1y1[1]:x2y2[1],x1y1[0]:x2y2[0]]
            #customization
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            fgmask = fgbg.apply(gray)
            thresh = 15
            maxValue = 40
            ret, th1 = cv2.threshold(fgmask, thresh, maxValue, cv2.THRESH_TRUNC)
            accum_image = cv2.add(accum_image, th1)
            #essentials
           # if class_names[int(classes[i])]=='person' :
            #    img = cv2.rectangle(img, x1y1, x2y2, (255, 255, 0), 1)
            crop_img=img[x1y1[1]:x2y2[1],x1y1[0]:x2y2[0]]
            crop_black=accum_image[x1y1[1]:x2y2[1],x1y1[0]:x2y2[0]]
            if class_names[int(classes[i])]=='crowd' :
                img = cv2.rectangle(img, x1y1, x2y2, (0,0 , 255), 1)
                
                
                img = cv2.putText(img, '{} {:.2f}'.format(
                class_names[int(classes[i])], objectness[i]),
                (x1y1[0],x2y2[1]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0,0 ), 1)
                color_image = im_color = cv2.applyColorMap(crop_black, cv2.COLORMAP_HOT)
                img[x1y1[1]:x2y2[1],x1y1[0]:x2y2[0]]= cv2.addWeighted(crop_img, 0.5, color_image, 0.5, 0)
            #img= cv2.addWeighted(img, 0.5, color_image, 0.5, 0)
        #cv2.imshow('result over',img)
    return img


def draw_labels(x, y, class_names):
    img = x.numpy()
    boxes, classes = tf.split(y, (4, 1), axis=-1)
    classes = classes[..., 0]
    wh = np.flip(img.shape[0:2])
    for i in range(len(boxes)):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        img = cv2.rectangle(img, x1y1, x2y2, (0, 255, 0), 1)
       # img = cv2.putText(img, class_names[classes[i]],
        #                  x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL,
         #                 1, (0, 0, 255), 2)
    return img


def freeze_all(model, frozen=True):
    model.trainable = not frozen
    if isinstance(model, tf.keras.Model):
        for l in model.layers:
            freeze_all(l, frozen)
