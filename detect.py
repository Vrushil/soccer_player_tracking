import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from playertracking.models import (
    YoloV3, YoloV3Tiny,setmethod
)
from playertracking.dataset import transform_images, load_tfrecord_dataset
from playertracking.utils import draw_outputs

flags.DEFINE_string('classes', './data/soccerv2.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3_train_24.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('image', './data/socgirl.jpg', 'path to input image')
flags.DEFINE_string('tfrecord', None, 'tfrecord instead of image')
flags.DEFINE_string('output', './output.jpg', 'path to output image')
flags.DEFINE_integer('num_classes', 10, 'number of classes in the model')


def main(_argv):
    

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))    
    
    
    yolo_player_classes='./data/coco.names'
    yolo_player= YoloV3(classes=80)
    yolo_player.load_weights('./checkpoints/yolov3.tf')
    
    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights).expect_partial()
    logging.info('weights loaded')
    yolo_player_class_names=[c.strip() for c in open(yolo_player_classes).readlines()]
    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    if FLAGS.tfrecord:
        dataset = load_tfrecord_dataset(
            FLAGS.tfrecord, FLAGS.classes, FLAGS.size)
        dataset = dataset.shuffle(512)
        img_raw, _label = next(iter(dataset.take(1)))
    else:
        img_raw = tf.image.decode_image(
            open(FLAGS.image, 'rb').read(), channels=3)

    img = tf.expand_dims(img_raw, 0)
    img = transform_images(img, FLAGS.size)

    t1 = time.time()
    setmethod(1)
    p_boxes, p_scores, p_classes, p_nums = yolo_player.predict(img)
    setmethod(0)
    boxes, scores, classes, nums = yolo.predict(img)
    t2 = time.time()
    logging.info('time: {}'.format(t2 - t1))

    logging.info('detections:')
    for i in range(nums[0]):
        logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                           np.array(scores[0][i]),
                                           np.array(boxes[0][i])))

    img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
    img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
    img= draw_outputs(img,( p_boxes, p_scores, p_classes, p_nums),yolo_player_class_names)
    cv2.imwrite(FLAGS.output, img)
    cv2.imshow("op",img)
    logging.info('output saved to: {}'.format(FLAGS.output))


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
