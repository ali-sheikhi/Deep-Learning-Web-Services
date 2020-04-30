# Deep Learning Web Services

This is a simple project that allows one to host deep learning services like image classification, image object detection and video object detection on a local machine using Flask. Both types of tasks use pretrained models; VGG16 for classificatin and YOLOv3 for detection.

## Getting Started

Get your environment running (preferrably a new one to avoid package version conflicts) and navigate to the `flask_apps` folder(hencforth referred to as `project directory`).

### Prerequisites

All required packages are listed in [requirements.txt](requirements.txt). Run the following command to install them.

```
pip install -r requirement.txt
```

### Pre-trained Models

You will need to download the pre-trained models for the classification and object-detection tasks.

#### Classification Pre-trained Model Source

Download the [VGG_16_cat_and_dogs](https://drive.google.com/uc?id=19yICdtSbU_YkQBRxJ2if9KJwUL1oY5xs&export=download) pre-trained model into a new folder `VGG16` within the project directory, i.e. `./VGG16/`. 

#### Object-detection Pre-trained Model Source

Download the weights and and configuration files for the model `YOLOv3-320` from [YOLOv3](https://pjreddie.com/darknet/yolo/) into the existing folder `Yolo_v3` within the project directory, i.e. `./Yolo_v3/`. This model has been trained on the `COCO` dataset the labels (pertaining to 80 common objects) for which are already in the file `coco.names` in the folder `Yolo_v3`.

**Note**: The model is trained on images of different sizes but this being a web service for which speed is a priority `YOLOv3-320` is recommended.


## Testing It Out

You can either make predictions directly from the command line or through the browser.

### Command-line Predictions

Images or videos can be specified or one may leave out the path argument and the model will use a default.

#### Classification

Run the following command:

```
python VGG16c_d.py --image_path C:\your\path\here\example.jpg
```

#### Object-Detection

* Run the following command for `image` input:

```
python yolo.py --image 1 --image_path C:\your\path\here\example.jpg
```

* Run the following command for `video` (mp4) input:

```
python yolo.py --play_video 1 --video_path C:\your\path\here\example.mp4
```

* Run the following command for `camera feed` input:

```
python yolo.py --webcam 1
```


### Browser Predictions

In order to get the backend flask app running, run these

```
set FLASK_APP=predict_app.py
flask run
```
The server has been set to localhost in the script.

Then open your browser (testing was done on Chrome browser) and go to `http://localhost:5000`. Running the different tasks is pretty intuitive. For image and video object detection, once output is ready a save file dialog box will pop up (though for the former the processed image will also be displayed in the browser itself). So the bigger the mp4 the longer the wait.

Note: Webcam functionality is not available through browser.


## Acknowledgments

* [DeepLizard](https://www.youtube.com/channel/UC4UJ26WkceqONNF5S26OiVw) : Flask setup video series
* [nandinib1999](https://github.com/nandinib1999/object-detection-yolo-opencv) : Github page for YOLOV3 object detection using opencv
