# [How to Detect Human Faces in Python using OpenCV](https://www.thepythoncode.com/article/detect-faces-opencv-python)

## To run this

- `pip3 install -r requirements.txt`
- If you want to detect faces in a sample image like `me.jpg`, run `images-face-detect.py` script that generated another image `me.jpg` which contains rectangles around detected faces.
- If you want to detect faces in your live cam, run `live-face-detect.py`

## Output in images

  ![me.jpg](output/me_dnn_detected.jpg)

## Path prototxt model Caffe

```python
# https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
prototxt_path = "weights/deploy.prototxt.txt"
```

## Path model Caffe

```python
# https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel
model_path = "weights/res10_300x300_ssd_iter_140000_fp16.caffemodel"
```
