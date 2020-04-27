import base64
import io
import os
from PIL import Image
from flask import request, send_file, jsonify, Flask, render_template, send_from_directory, Response, stream_with_context
from tensorflow.python.keras.backend import set_session
from gevent.pywsgi import WSGIServer
import cv2

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

img_dir = r'.\static\images_temp'
test_img_name = 'test.jpg'
test_img_path = os.path.join(img_dir, test_img_name)
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
    _,encoded_rendered_image = cv2.imencode('.png', res)
    ENCODING = 'utf-8'
    encoded_rendered_image_text = base64.b64encode(encoded_rendered_image)
    encoded_rendered_base64_string = encoded_rendered_image_text.decode(ENCODING)
    response = {"prediction": {"rendered_image":encoded_rendered_base64_string}}
    cv2.imwrite(test_img_path, res)
    return jsonify(response)


@app.route("/Yolov3DI", methods=["POST","GET"])
def imgDL():
    return send_from_directory(img_dir, test_img_name, as_attachment=True)



 
#=========================================
# Yolov3 - Videos


vid_dir = r'.\static\videos_temp'
test_vid_name = 'test.mp4'
processed_vid_name = 'processed.mp4'
test_vid_path = os.path.join(vid_dir, test_vid_name)
processed_vid_path = os.path.join(vid_dir, processed_vid_name)

test_frame_name = 'frame.jpg'
test_frame_path = os.path.join(img_dir, test_frame_name)

model, classes, colors, output_layers=yl3.get_model(cfgpath,wpath,labelspath)
print(" * Yolo_v3 Object Detector model loaded!")


def stream_template(template_name, **context):
    app.update_template_context(context)
    t = app.jinja_env.get_template(template_name)
    rv = t.stream(context)
    rv.enable_buffering(5)
    return rv


@app.route("/Yolov3DetectV", methods=["POST","GET"])
def predictYoloV():
    message = request.get_json(force=True)
    encoded = message['video']
    decoded = base64.b64decode(encoded + '===')
    with open(test_vid_path, 'wb') as wfile:
        wfile.write(decoded)
    cap = cv2.VideoCapture(test_vid_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(processed_vid_path, cv2.VideoWriter_fourcc(*'MP4V'), 10, (frame_width,frame_height))
    _, frame = cap.read()
    while _:
        height, width, channels = frame.shape
        blob, outputs = yl3.detect_objects(frame, model, output_layers)
        boxes, confs, class_ids = yl3.get_box_dimensions(outputs, height, width)
        res = yl3.draw_labels(boxes, confs, colors, class_ids, classes, frame)
        out.write(res)
        _, frame = cap.read()

    #def generateFrames():
    #    _, frame = cap.read()
    #    while _:
    #        height, width, channels = frame.shape
    #        blob, outputs = yl3.detect_objects(frame, model, output_layers)
    #        boxes, confs, class_ids = yl3.get_box_dimensions(outputs, height, width)
    #        res = yl3.draw_labels(boxes, confs, colors, class_ids, classes, frame)
    #        _,encoded_rendered_image = cv2.imencode('.png', res)
    #        ENCODING = 'utf-8'
    #        encoded_rendered_image_text = base64.b64encode(encoded_rendered_image)
    #        encoded_rendered_base64_string = encoded_rendered_image_text.decode(ENCODING)
    #        response = {"prediction": {"rendered_image":encoded_rendered_base64_string}}
    #        out.write(res)
    #        _, frame = cap.read()
    #        yield jsonify(response)
            
    response = {"status": "ok"}
    #cap.release()
    #out.release()
    #return Response(stream_template('predict.html', frames = stream_with_context(generateFrames)))
    return jsonify(response)


@app.route("/Yolov3DV", methods=["POST","GET"])
def vidDL():
    return send_from_directory(vid_dir, processed_vid_name, as_attachment=True)







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