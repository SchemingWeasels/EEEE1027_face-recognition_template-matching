#01 Accessing Webcam 

import cv2
cap = cv2.VideoCapture(0)
# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5,interpolation=cv2.INTER_AREA)
    cv2.imshow('Input', frame)
    c = cv2.waitKey(1)

    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
________________________________________________________________________
#02. Using different colour space

import cv2

def print_howto():
	print("""Change color space of video stream
		1. Grayscale - press 'g'
		2. YUV - press 'y'
		3. HSV - press 'h' """)

if __name__=='__main__':
	print_howto()
	cap = cv2.VideoCapture(0)
# Check if the webcam is opened correctly
	if not cap.isOpened():
		raise IOError("Cannot open webcam")

	c_mode = None
	
	while True:
# Read the current frame from webcam
		ret, frame = cap.read()
# Resize the captured image
		frame = cv2.resize(frame, None, fx=0.5, fy=0.5,
		interpolation=cv2.INTER_AREA)
		c = cv2.waitKey(1)
		if c == 27:
			break
# Update c_mode only in case it is different and key was pressed
# In case a key was not pressed during the iteration result is -1or 255, depending
# on library versions
		if c != -1 and c != 255 and c != cur_mode:
			c_mode = c

		if c_mode == ord('g'):
			output = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		elif c_mode == ord('y'):
			output = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
		elif c_mode == ord('h'):
			output = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		else:
			output = frame
		cv2.imshow('Webcam', output)

cap.release()
cv2.destroyAllWindows()
_________________________________________________________________________
# Using Mouse in real time and capture region of interest (ROI)

import cv2
import numpy as np

def update_pts(params, x, y):
	global x_init, y_init
	params["top_left_pt"] = (min(x_init, x), min(y_init, y))
	params["bottom_right_pt"] = (max(x_init, x), max(y_init, y))
	img[y_init:y, x_init:x] = 255 - img[y_init:y, x_init:x]

def draw_rectangle(event, x, y, flags, params):
	global x_init, y_init, drawing
# First click initialize the init rectangle point
	if event == cv2.EVENT_LBUTTONDOWN:
		drawing = True
		x_init, y_init = x, y
# Meanwhile mouse button is pressed, update diagonal rectangle point
	elif event == cv2.EVENT_MOUSEMOVE and drawing:
		update_pts(params, x, y)
# Once mouse botton is release
	elif event == cv2.EVENT_LBUTTONUP:
		drawing = False
		update_pts(params, x, y)

if __name__=='__main__':
	drawing = False
	event_params = {"top_left_pt": (-1, -1), "bottom_right_pt": (-1, -1)}
	cap = cv2.VideoCapture(0)
# Check if the webcam is opened correctly
	if not cap.isOpened():
		raise IOError("Cannot open webcam")
	
	cv2.namedWindow('Webcam')
# Bind draw_rectangle function to every mouse event
	cv2.setMouseCallback('Webcam', draw_rectangle, event_params)
	
	while True:
		ret, frame = cap.read()
		img = cv2.resize(frame, None, fx=0.5, fy=0.5,interpolation=cv2.INTER_AREA)
		(x0,y0), (x1,y1) = event_params["top_left_pt"],event_params["bottom_right_pt"]
		img[y0:y1, x0:x1] = 255 - img[y0:y1, x0:x1]
		cv2.imshow('Webcam', img)
		c = cv2.waitKey(1)

		if c == 27:
			break
			
cap.release()
cv2.destroyAllWindows()
________________________________________________________________________
#05 .. Using Morphology Operation and basic filtering

import cv2
import numpy as np

def print_howto():
	print(""" Cartoonizing mode of image:
			1. Cartoonize without Color - press 's'
			2. Cartoonize with Color - press 'c' """)

def cartoonize_image(img, ksize=5, sketch_mode=False):
	num_repetitions, sigma_color, sigma_space, ds_factor = 10, 5, 7, 4
# Convert image to grayscale
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Apply median filter to the grayscale image
	img_gray = cv2.medianBlur(img_gray, 7)
# Detect edges in the image and threshold it
	edges = cv2.Laplacian(img_gray, cv2.CV_8U, ksize=ksize)
	ret, mask = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY_INV)
# 'mask' is the sketch of the image
	if sketch_mode:
		img_sketch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
		kernel = np.ones((3,3), np.uint8)
		img_eroded = cv2.erode(img_sketch, kernel, iterations=1)
		return cv2.medianBlur(img_eroded, ksize=5)
		
# Resize the image to a smaller size for faster computation
	img_small = cv2.resize(img, None, fx=1.0/ds_factor, fy=1.0/ds_factor, interpolation=cv2.INTER_AREA)
# Apply bilateral filter the image multiple times
	for i in range(num_repetitions):
		img_small = cv2.bilateralFilter(img_small, ksize, sigma_color,sigma_space)
		
	img_output = cv2.resize(img_small, None, fx=ds_factor, fy=ds_factor,interpolation=cv2.INTER_LINEAR)
	dst = np.zeros(img_gray.shape)
# Add the thick boundary lines to the image using 'AND' operator
	dst = cv2.bitwise_and(img_output, img_output, mask=mask)
	return dst

if __name__=='__main__':
	print_howto()
	cap = cv2.VideoCapture(0)
	cur_mode = None

	while True:
		ret, frame = cap.read()
		frame = cv2.resize(frame, None, fx=0.5, fy=0.5,interpolation=cv2.INTER_AREA)
		c = cv2.waitKey(1)
		if c == 27:
			break

		if c != -1 and c != 255 and c != cur_mode:
			cur_mode = c
		
		if cur_mode == ord('s'):
			cv2.imshow('Cartoonize', cartoonize_image(frame, ksize=5, sketch_mode=True))
		elif cur_mode == ord('c'):
			cv2.imshow('Cartoonize', cartoonize_image(frame, ksize=5,sketch_mode=False))
		else:
			cv2.imshow('Cartoonize', frame)

cap.release()
cv2.destroyAllWindows()
________________________________________________________________________
#05a... Smooting filter with bilateral filter
import cv2
import numpy as np
img = cv2.imread('Pictures/Carpettiles.jpg') #replace with your own picture
img_gaussian = cv2.GaussianBlur(img, (13,13), 0) # Gaussian Kernel Size 13x13
img_bilateral = cv2.bilateralFilter(img, 13, 70, 50)
cv2.imshow('Input', img)
cv2.imshow('Gaussian filter', img_gaussian)
cv2.imshow('Bilateral filter', img_bilateral)
cv2.waitKey()
cv2.destroyAllWindows() 

_______________________________________________________________________
#05b... Median filter to reduce noise
import cv2
import numpy as np
img = cv2.imread('Pictures/images2.jpg') #replace with your own picture
output = cv2.medianBlur(img, ksize=7)
cv2.imshow('Input', img)
cv2.imshow('Median filter', output)
cv2.waitKey()
cv2.destroyAllWindows()
