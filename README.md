# tvm_yolo
## Install tvm from:
```
https://tvm.apache.org/docs/install/from_source.html#install-from-source
```
## Download yolov5
```
# Clone YOLOv5
git clone https://github.com/ultralytics/yolov5
cd yolov5
# Install required packages
pip install -r requirements.txt
```
## Execution
- Example: 
```
python yolo_test.py --img input.jpg --out output.jpg --imgsz 640
```
- Another instruction: 
  - `--enable_optimizing 1`: allow optimizing
  - `--no-run`: Don't run the inference for input.jpg
 
- Output:
  - `output.jpg`: output image
  - `output.log`: Relax IR module of best.pt
