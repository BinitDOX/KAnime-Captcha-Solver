from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from PIL import Image
from threading import Thread
import time

from keras.models import load_model
from sklearn.preprocessing import LabelEncoder

import os
import cv2
import numpy as np
import tensorflow as tf
import sys

sys.path.append("F:/Tensorflow1/models")
sys.path.append("F:/Tensorflow1/models/slim")
sys.path.append("F:/Tensorflow1/models/research/object_detection")

from utils import label_map_util

#--Globals--
START_TIME = 0
END_TIME = 0
IMG_SIZE = 150
N = 0
Z = []
DIR = 'Temp/'
MODEL_PATH_DIGIT = 'Models/Digit/'
MODEL_PATH_CONTENT = 'Models/Content/modelContent.h5'
CWD_PATH = os.getcwd()
CKPT_PATH = os.path.join(CWD_PATH,MODEL_PATH_DIGIT,'frozen_inference_graph.pb')
LABEL_PATH = os.path.join(CWD_PATH,MODEL_PATH_DIGIT,'labelmap.pbtxt')
EXTENSION_PATH = os.path.join(CWD_PATH, 'Extensions/ublock_1_27_10_0.crx')
CAPTCHA_URL = 'https://kissanime.ru/Special/AreYouHuman2?'
CAP_CONTENT_TYPES = ['arm', 'bird', 'bear', 'boy', 'burger', 'cat','cloud', 'couple', 'cup', 'dice', 'dog', 'emoticon', 'girl', 'green', 'hand', 'lion', 'medal', 'penguin', 'rabbit', 'tiger', 'turtle', 'yellow', 'sheep']


#--Selenium--
chromeOptions = Options()
chromeOptions.add_argument("user-data-dir=ChromeUserData")
chromeOptions.add_extension(EXTENSION_PATH)
driver = webdriver.Chrome(ChromeDriverManager().install(), chrome_options=chromeOptions)
driver.get("http://kissanime.ru/")


def getCaptchaImages():
    global imgElems
    
    driver.maximize_window()
    driver.execute_script("window.scrollTo(0,0);")
    imgElems = driver.find_elements_by_xpath('//img[@indexvalue]')
    imgElemLocs = [img.location for img in imgElems]
    imgElemSizes = [img.size for img in imgElems]
    
    locLefts = [loc['x'] for loc in imgElemLocs]
    locTops = [loc['y'] for loc in imgElemLocs]
    locRights = [loc['x'] + size['width'] for (loc, size) in zip(imgElemLocs, imgElemSizes)]
    locBotts = [loc['y'] + size['height'] for (loc, size) in zip(imgElemLocs, imgElemSizes)]

    driver.save_screenshot('Temp.png')
    fullImg = Image.open('Temp.png')
    images = [fullImg.crop((left, top, right, bottom)) for (left, top, right, bottom) in zip(locLefts, locTops, locRights, locBotts)] 
    [image.save(DIR + 'img{}.png'.format(images.index(image)+1)) for image in images]


def getCaptchaQuery():
    global query
    queries = driver.find_elements_by_xpath('//span')
    query1 = queries[0].get_attribute("innerText").split(', ')
    query2 = queries[1].get_attribute("innerText").split(', ')
    query = [query1, query2]
	
	
def detectCaptchaPageThread():
	global END_TIME, START_TIME
	while True:
		url = driver.current_url
		if CAPTCHA_URL in url:
			try:
				print('\n(+) Detected Captcha page')
				START_TIME = time.process_time()
				getCaptchaImages()
				getCaptchaQuery()
				END_TIME = time.process_time()
				print('(+) Took %.4s sec to fetch Captcha assets' % (END_TIME-START_TIME))
				
				main()

			except:
				print("\n(!) Error: ", sys.exc_info()[0])
				print("\n(-) Thread is now on cooldown of 20sec")
				time.sleep(20)
				print("\n(+) Thread restarted")
				
		time.sleep(0.5)


def sendClick(index):
	imgElems[index].click()

def main():
	global END_TIME, START_TIME
	START_TIME = time.process_time()
	digitPreds = predictDigit()
	contentPreds = predictContent(DIR)
	
	for i in range(0, 2):
		index = contentPreds.index(query[i][0])
		if digitPreds[index] == query[i][2]:
			sendClick(index)
		elif digitPreds[index+1] == query[i][2]:
			sendClick(index+1)
		else:
			index = contentPreds[index+2:].index(query[i][0]) + index+2
			if digitPreds[index] == query[i][2]:
				sendClick(index)
			elif digitPreds[index+1] == query[i][2]:
				sendClick(index+1)
	END_TIME = time.process_time()
	print('(+) Took %.4s sec to execute' % (END_TIME-START_TIME))


def loadDigitPredictor():
	global session
	global image_tensor, detection_boxes, detection_scores, detection_classes, num_detections
	detection_graph = tf.Graph()
	with detection_graph.as_default():
		od_graph_def = tf.GraphDef()
		with tf.gfile.GFile(CKPT_PATH, 'rb') as fid:
			serialized_graph = fid.read()
			od_graph_def.ParseFromString(serialized_graph)
			tf.import_graph_def(od_graph_def, name='')
			config = tf.ConfigProto()
			config.gpu_options.allow_growth = True
	
		session = tf.Session(graph=detection_graph, config=config)
		
	image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
	detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
	detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
	detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
	num_detections = detection_graph.get_tensor_by_name('num_detections:0')

def predictDigit():
	digits = []
	tempDIR = 'Temp'
	files = os.listdir(DIR)
	if(len(files)<8):
		imagePaths = [ os.path.join(tempDIR, 'img{}.png'.format(i)) for i in range(1, 7) ]
	else:
		imagePaths = [ os.path.join(tempDIR, 'img{}.png'.format(i)) for i in range(1, 9) ]
	for imagePath in imagePaths:
		image = cv2.imread(imagePath)
		image_expanded = np.expand_dims(image, axis=0)
		(boxes, scores, classes, num) = session.run(
			[detection_boxes, detection_scores, detection_classes, num_detections],
			feed_dict={image_tensor: image_expanded})
		digits.append(str(int(classes[0, 0] - 1)))
	return digits

def loadContentPredictor():
	global modelContent, labelEncoder
	modelContent = load_model(os.path.join(CWD_PATH, MODEL_PATH_CONTENT))
	modelContent._make_predict_function()
	
	for c_type in CAP_CONTENT_TYPES:
		label = c_type
		Z.append(str(label))
	labelEncoder = LabelEncoder()
	labelEncoder.fit_transform(Z)
	
def predictContent(DIR):
	contents = []
	for img in os.listdir(DIR):
		path = os.path.join(DIR, img)
		img = cv2.imread(path, cv2.IMREAD_COLOR)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
		contents.append(np.array(img))
	contents = np.array(contents)
	contents = contents/255
	
	pred = modelContent.predict(contents)
	pred_temp = np.argmax(pred ,axis=1)
	content_preds = labelEncoder.inverse_transform(pred_temp).tolist()
	return content_preds



if __name__ ==  '__main__':
	print('\n(-) Initializing...\n')
	loadDigitPredictor()
	print('\n(+) Loaded digit predictor')
	loadContentPredictor()
	print('\n(+) Loaded content predictor')
	
	print('\n(+) Initialized & Running...')
	thread = Thread(target = detectCaptchaPageThread)
	thread.start()
