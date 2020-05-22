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

conda env create -f conda-cpu.yml

conda activate soccer-cpu


###### Tensorflow GPU

conda env create -f conda-gpu.yml

conda activate soccer-gpu

#### 2. Downloading weights

- Download the trained weights from [here](https://drive.google.com/drive/folders/1klC5txyxosWLWG3raIm5ur04F807uUaG?usp=sharing) 
- Move the downloaded weights and files to checkpoints/ 

## Detections
 
 ###### for image detection
 
 python detect.py  --image   ./data/socgirl.jpg
 
 ###### for video
 
 python detect_video.py  --video  ./data/goals.mp4
 
## References

##### [YOLOv3 implemented in TensorFlow 2.0](https://github.com/zzh8829/yolov3-tf2) by [Zihao Zhang](https://github.com/zzh8829)
