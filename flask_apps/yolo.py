import cv2
import numpy as np 
import argparse
import time
import os


# Set paths for dataset classes, model weights and config
labelspath= os.path.join(os.path.dirname(__file__),"Yolo_v3","coco.names")
cfgpath= os.path.join(os.path.dirname(__file__),"Yolo_v3","yolov3.cfg")
wpath= os.path.join(os.path.dirname(__file__),"Yolo_v3","yolov3.weights")

#Load yolo
def get_model():
	net = cv2.dnn.readNet(cfgpath, wpath)
	classes = []
	with open(labelspath, "r") as f:
		classes = [line.strip() for line in f.readlines()]

	layers_names = net.getLayerNames()
	output_layers = [layers_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
	colors = np.random.uniform(0, 255, size=(len(classes), 3))
	return net, classes, colors, output_layers

def load_image(img_path):
	# image loading
    img = cv2.imread(img_path)
    npimg=np.array(img)
    image=npimg.copy()
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    img= cv2.resize(image, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape
    return img, height, width, channels

def load_image_flask(img):
	# image loading
	#img = cv2.imread(img_path)
    npimg=np.array(img)
    image=npimg.copy()
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    img= cv2.resize(image, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape
    return img, height, width, channels

def start_webcam():
	cap = cv2.VideoCapture(0)

	return cap


def display_blob(blob):
	'''
		Three images each for RED, GREEN, BLUE channel
	'''
	for b in blob:
		for n, imgb in enumerate(b):
			cv2.imshow(str(n), imgb)

def detect_objects(img, net, outputLayers):			
	blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
	net.setInput(blob)
	outputs = net.forward(outputLayers)
	return blob, outputs

def get_box_dimensions(outputs, height, width):
	boxes = []
	confs = []
	class_ids = []
	for output in outputs:
		for detect in output:
			scores = detect[5:]
			class_id = np.argmax(scores)
			conf = scores[class_id]
			if conf > 0.3:
				center_x = int(detect[0] * width)
				center_y = int(detect[1] * height)
				w = int(detect[2] * width)
				h = int(detect[3] * height)
				x = int(center_x - w/2)
				y = int(center_y - h / 2)
				boxes.append([x, y, w, h])
				confs.append(float(conf))
				class_ids.append(class_id)
	return boxes, confs, class_ids

def draw_labels(boxes, confs, colors, class_ids, classes, img): 
	indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
	font = cv2.FONT_HERSHEY_PLAIN
	for i in range(len(boxes)):
		if i in indexes:
			x, y, w, h = boxes[i]
			label = str(classes[class_ids[i]]) + (":%.2f" % confs[i])
			color = colors[i]
			cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
			cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
	cv2.imshow("Image", img)

def draw_labels_flask(boxes, confs, colors, class_ids, classes, img): 
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
	    if i in indexes:
		    x, y, w, h = boxes[i]
		    label = str(classes[class_ids[i]]) + (":%.2f" % confs[i])
		    color = colors[i]
		    cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
		    cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
    #cv2.imshow("Image", img)
    return img

def image_detect(img_path): 
	model, classes, colors, output_layers = get_model()
	image, height, width, channels = load_image(img_path)
	blob, outputs = detect_objects(image, model, output_layers)
	boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
	draw_labels(boxes, confs, colors, class_ids, classes, image)
	while True:
		key = cv2.waitKey(1)
		if key == 27:
			break

def image_detect_flask(model, classes, colors, output_layers, img): 
	#model, classes, colors, output_layers = get_model()
    image, height, width, channels = load_image_flask(img)
    blob, outputs = detect_objects(image, model, output_layers)
    boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
    res = draw_labels_flask(boxes, confs, colors, class_ids, classes, image)
# =============================================================================
#     while True:
# 	    key = cv2.waitKey(1)
# 	    if key == 27:
# 		    break
# =============================================================================
    return res

def webcam_detect():
	model, classes, colors, output_layers = get_model()
	cap = start_webcam()
	while True:
		_, frame = cap.read()
		height, width, channels = frame.shape
		blob, outputs = detect_objects(frame, model, output_layers)
		boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
		draw_labels(boxes, confs, colors, class_ids, classes, frame)
		key = cv2.waitKey(1)
		if key == 27:
			break
	cap.release()


def start_video(video):
    model, classes, colors, output_layers = get_model()
    cap = cv2.VideoCapture(video)
    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
	# Define the fps to be equal to 10. Also frame size is passed.
    # frame_width = int(cap.get(3))
    # frame_height = int(cap.get(4))
    # out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
    while True:
	    _, frame = cap.read()
	    height, width, channels = frame.shape
	    blob, outputs = detect_objects(frame, model, output_layers)
	    boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
	    draw_labels(boxes, confs, colors, class_ids, classes, frame)
	    key = cv2.waitKey(1)
	    if key == 27:
		    break
    cap.release()




if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--webcam', help="True/False", default=False)
	parser.add_argument('--play_video', help="Tue/False", default=False)
	parser.add_argument('--image', help="Tue/False", default=False)
	parser.add_argument('--video_path', help="Path of video file", default="Test media - object detection/videos/car_on_road.mp4")
	parser.add_argument('--image_path', help="Path of image to detect objects", default="Test media - object detection/Images/bicycle.jpg")
	parser.add_argument('--verbose', help="To print statements", default=True)
	args = parser.parse_args()
	webcam = args.webcam
	video_play = args.play_video
	image = args.image
	if webcam:
		if args.verbose:
			print('---- Starting Web Cam object detection ----')
		webcam_detect()
	if video_play:
		video_path = args.video_path
		if args.verbose:
			print('Opening '+video_path+" .... ")
		start_video(video_path)
	if image:
		image_path = args.image_path
		if args.verbose:
			print("Opening "+image_path+" .... ")
		image_detect(image_path)
	

	cv2.destroyAllWindows()
