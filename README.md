# Soccer_player_tracking using Tensorflow 2.0

## YOLOv3 Object Detection Model trained on Custom dataset  
An implementation to track Soccer players on any random video. The model also detects background  such as crowd and hoardings.

## Project Requirements

- Python 3.7
- TensorFlow 2.0
- OpenCV
- Numpy
- Matplotlib
- Lxml
- tqdm
- Cudnn
- Cuda toolkit

## Usage
#### 1. Creating an environment for the repository using conda(Installation)
##### Tensorflow CPU

conda env create -f conda-cpu.yml   // To create an environment where tensorflow-gpu is not supported

conda activate soccer-cpu


###### Tensorflow GPU

conda env create -f conda-gpu.yml  // Creating an environment for tensorflow-gpu

conda activate soccer-gpu

#### 2. Downloading weights

- Download the trained weights from [here](https://drive.google.com/drive/folders/1klC5txyxosWLWG3raIm5ur04F807uUaG?usp=sharing) 
- Move the downloaded weights and files to checkpoints/ 

## Detections
 
 ###### for image detection
 
 python detect.py  --image   ./data/socgirl.jpg  // for detections in an image
 
 ###### for video
 
 python detect_video.py  --video  ./data/goals.mp4 // for detections in a video file
 
 ## File Details
 
 convert.py // Convert the weights of YOLOv3 to  .tf format
 
 train.py  // to train your own model using custom dataset
 
 utils.py // Draws outputs onto the image/frame using information received from model
 
 models.py // All the model functions are in here.
 
 coco.names // Contains the class names of the COCO dataset
 
 soccerv2.names //  Contains the class names of our custom trained model
 
## References

##### [YOLOv3 implemented in TensorFlow 2.0](https://github.com/zzh8829/yolov3-tf2) by [Zihao Zhang](https://github.com/zzh8829)
