import base64
import io
import os
from PIL import Image
from flask import request, send_file, jsonify, Flask, render_template, send_from_directory
#from flask import Response, make_response
from tensorflow.python.keras.backend import set_session
from gevent.pywsgi import WSGIServer
import cv2
#import random
#import string

import VGG16c_d as vg # methods for dog and cat classification
import yolo as yl3    # methods for yolov3 object detection

app = Flask(__name__)
app.config["DEBUG"] = True


@app.route("/")
def index():
    return render_template('predict.html')


#=========================================
# dog and cat classification

print(" * Loading cat/dog classification model...")
sess = vg.initialize_sess()
set_session(sess)
global model, graph
wpath='VGG16\VGG16_cats_and_dogs.h5'
model_vgg, graph_vgg = vg.get_model(wpath)
print(" * Cat/dog model loaded!")

@app.route("/catORdog", methods=["POST","GET"])
def predictCorD():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded + '===')
    #decoded = base64.b64decode(encoded)

    with graph_vgg.as_default():
        set_session(sess)
        image = Image.open(io.BytesIO(decoded))
        processed_image = vg.preprocess_image(image, target_size=(224, 224))
        prediction = model_vgg.predict(processed_image).tolist()

    response = {
        'prediction': {
            'dog': prediction[0][0],
            'cat': prediction[0][1]
        }
    }
    return jsonify(response)


#=========================================
# Yolov3 for object detection of 80 common objects
# Set paths for dataset classes, model weights and config


labelspath="Yolo_v3/coco.names"
cfgpath="Yolo_v3/yolov3.cfg"
wpath="Yolo_v3/yolov3.weights"


#=========================================
# Yolov3 - Images

test_img_path = r'.\static\images_temp'
test_img_name = 'test.jpg'
test_img_dir = os.path.join(test_img_path, test_img_name)
model, classes, colors, output_layers=yl3.get_model(cfgpath,wpath,labelspath)
print(" * Yolo_v3 Object Detector model loaded!")
# =============================================================================
# def randomString(stringLength=10):
#     """Generate a random string of fixed length """
#     letters = string.ascii_lowercase
#     return ''.join(random.choice(letters) for i in range(stringLength))
# =============================================================================

@app.route("/Yolov3DetectI", methods=["POST","GET"])
def predictYoloI():
    message = request.get_json(force=True)
    #global filename 
    #filename = "{}.jpg".format(randomString())
    encoded = message['image']
    decoded = base64.b64decode(encoded + '===')
    image = Image.open(io.BytesIO(decoded))
    res = yl3.image_detect(model, classes, colors, output_layers, image)
    cv2.imwrite(test_img_dir, res)
    #img = cv2.imread(os.path.join(test_image_path, tmp_img))
    #_, im_arr = cv2.imencode('.jpg', img)  # im_arr: image in Numpy one-dim array format.
    #response = make_response(im_arr.tobytes())
    #response.headers['Content-Type'] = 'image/png'
    #im_b64 = base64.b64encode(im_bytes)
    #encoded_img = "data:image/jpeg;base64," + (im_b64).decode("utf-8")
# =============================================================================
#     response = {
#         'prediction': {
#             'image': im_b64
#         }
#     }
# =============================================================================
    # for use with curl in cmd: curl.exe -X POST -F image=@test.jpg 'http://localhost:5000/Yolov3Detect' --output test.png
    #return Response(response=res, status=200, mimetype="image/jpeg")

    #return send_file(imgByteArr, mimetype='image/png', as_attachment=False)
    return render_template('predict.html')


@app.route("/Yolov3DI", methods=["POST","GET"])
def imgDL():
    return send_from_directory(test_img_path, test_img_name, as_attachment=True)


 
#=========================================
# Yolov3 - Videos


test_vid_path = r'.\static\videos_temp'
test_vid_name = 'test.mp4'
test_frame_path = r'.\static\images_temp'
test_vid_dir = os.path.join(test_vid_path, test_vid_name)
test_frame_name = 'test.jpg'
test_frame_dir = os.path.join(test_vid_path, test_frame_name)
model, classes, colors, output_layers=yl3.get_model(cfgpath,wpath,labelspath)
print(" * Yolo_v3 Object Detector model loaded!")

def vidFeed():
    return render_template('predict.html')

@app.route("/Yolov3DetectV", methods=["POST","GET"])
def predictYoloV():
    message = request.get_json(force=True)
    encoded = message['video']
    decoded = base64.b64decode(encoded + '===')
    with open(test_vid_dir, 'wb') as wfile:
        wfile.write(decoded)
    cap = cv2.VideoCapture(test_vid_dir)
    _, frame = cap.read()
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(test_vid_dir, cv2.VideoWriter_fourcc(*'MP4V'), 10, (frame_width,frame_height))
    while _:
        height, width, channels = frame.shape
        blob, outputs = yl3.detect_objects(frame, model, output_layers)
        boxes, confs, class_ids = yl3.get_box_dimensions(outputs, height, width)
        res = yl3.draw_labels(boxes, confs, colors, class_ids, classes, frame)
        out.write(res)
        cv2.imwrite(test_frame_dir, res)
        vidFeed()
        _, frame = cap.read()
    cap.release()
    out.release()
    #image = Image.open(io.BytesIO(decoded))


@app.route("/Yolov3DV", methods=["POST","GET"])
def vidDL():
    return send_from_directory(test_vid_path, test_vid_name, as_attachment=True)







# =============================================================================
# @app.after_request
# def add_header(response):
#     response.headers['Pragma'] = 'no-cache'
#     response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, public, max-age=0'
#     response.headers['Expires'] = '0'
#     return response
# =============================================================================


# =============================================================================
# @app.errorhandler(404)
# def page_not_found(e):
#     return jsonify(error=404, text=str(e)), 404
# =============================================================================


if __name__ == '__main__':
   #app.run(host='0.0.0.0')
   
    # Serve the app with gevent
   http_server = WSGIServer(('', 5000), app)
   http_server.serve_forever()