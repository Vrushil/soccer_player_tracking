import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import tensorflow as tf
from playertracking.models import (
    YoloV3, YoloV3Tiny, YoloV32
)

from playertracking.dataset import transform_images
from playertracking.utils import draw_outputs,draw_outputs_3
import matplotlib.pyplot as plt


flags.DEFINE_string('classes', './data/soccerv2.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3_train_8.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video', './data/korea.mp4',
                    'path to video file or number for webcam)')
flags.DEFINE_string('output', './output.avi', 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 10, 'number of classes in the model')


def main(_argv):
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    #physical_devices = tf.config.experimental.list_physical_devices('GPU')
    #if len(physical_devices) > 0:
     #   tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
    yolo_player_classes='./data/coco.names'
    yolo_player= YoloV3(classes=80)
    #yolo_player_weights= str('./checkpoints/yolov3.tf')
    
    
    yolo_player.load_weights('./checkpoints/yolov3.tf')
    
    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)

    else:
        yolo = YoloV32(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')
    yolo_player_class_names=[c.strip() for c in open(yolo_player_classes).readlines()]
    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    times = []

    try:
        vid = cv2.VideoCapture(int(FLAGS.video))
    except:
        vid = cv2.VideoCapture(FLAGS.video)

    out = None

    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    while True:
        _, img = vid.read()

        if img is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            continue

        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, FLAGS.size)

        t1 = time.time()
       # setmethod(1)
        p_boxes, p_scores, p_classes, p_nums = yolo_player.predict(img_in)
        img=draw_outputs(img,( p_boxes, p_scores, p_classes, p_nums),yolo_player_class_names)
        
        #setmethod(0)
        boxes, scores, classes, nums = yolo.predict(img_in)
        img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
        
        #setmethod(1)
        t2 = time.time()
        times.append(t2-t1)
        times = times[-20:]

        #img = cv2.putText(img, "Time: {:.2f}ms".format(sum(times)/len(times)*1000), (0, 30),
         #            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
        if FLAGS.output:
            out.write(img)
        #cv2.imshow('output', img)
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
