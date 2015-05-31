#!/usr/bin/env python

''' - Card tracking and identification
'''

import numpy as np
import cv2
import time as t
import math, sys, os

# Resize scale factor for performance testing
resize = 0.7
debug = True

# Folder where image database is stored/written to.
train_folder = './trained_images/'

class Timer:
	def __init__(self):
		self.frames = 0
		self.start = t.time()

	def print_rate(self):
		diff = t.time() - self.start
		self.frames += 1

		# Reset counter every 1 second
		if (diff > 1):
			# Print out time and frame rate
			print "{0} {1}".format('FPS: ', (int)(self.frames / diff))
			self.start = t.time()
			self.frames = 0

def show_debug_img(title, img):
	if debug:
		cv2.waitKey(1)
		cv2.imshow(title, img)

def putText(img, text, pos):
	cv2.putText(img,text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))

# Maximum distance used by the FlannBasedMatcher
MAX_DISTANCE = 0.75
MIN_MATCH_COUNT = 40

class Stats():
	def __init__(self):
		self.found_card_strings = []

	def add_found_card(self, filename):
		self.found_card_strings.append(filename)

	def clear(self):
		self.found_card_strings = []

	def get(self):
		return " ".join(self.found_card_strings)

	def has_stats(self):
		return len(self.found_card_strings) > 0

# Here we perform matching against cards in the file database.
class Matching():

	def __init__(self, stats):
		self.folder = train_folder
		self.descriptors = {}
		self.stats = stats

	def test(self):
		img1 = cv2.imread(self.folder + 'trainedcard_85b.png')
		des1 = self.sift(img1, 'sift1')

		filelist = [f for f in os.listdir(self.folder) if f.endswith('.png')]
		for f in filelist:
			img2 = cv2.imread(self.folder + f)
			des2 = self.sift(img2, 'sift2')
			if self.flann_match(des1, des2):
				print 'found: {}'.format(self.folder + f)

	# Iterates through training files and builds sift features
	def detect_points(self):
		print 'Detecting features from file database..'
		filelist = [f for f in os.listdir(self.folder) if f.endswith('.png')]
		for f in filelist:
			img = cv2.imread(self.folder + f)
			self.descriptors[f] = self.sift(img, 'training')

	# Matching given image (img) against training files.
	def match(self, img):
		if len(self.descriptors) == 0:
			self.detect_points()
		des = self.sift(self.sharpen(img), 'sift1')
		return self.match_all(des)

	# Helper for match function, iterates all training file features.
	def match_all(self, des):
		for key in self.descriptors:
			src_des = self.descriptors[key]
			if self.flann_matcher(des, src_des):
				msg = 'found: {}'.format(key)
				stats.add_found_card(msg)
				print msg

	# Performs SIFT
	def sift(self, img, name):
		gray1 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		sift = cv2.SIFT()
		kp, des = sift.detectAndCompute(gray1,None)
		key1 = cv2.drawKeypoints(gray1,kp)
		show_debug_img(name, key1)
		return des

	# Brute force matcher (this isn't working)
	def bf_matcher(self,des1, des2):
		bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
		matches = bf.knnMatch(des1,des2, k=2)
		matches = sorted(matches, key = lambda x:x.distance)
		goodMatches = []
		#print 'matches: {}'.format(len(matches))

		''' Filter out matches based on the maximum distance parameter '''
		for m in matches:
			if len(m) == 2:
				if m[0].distance < (m[1].distance*MAX_DISTANCE):
					goodMatches.append(m[0])
		if len(goodMatches) < MIN_MATCH_COUNT:
			return False
		#print "Good matches: {}".format(len(goodMatches))
		return True

	# Flann matcher
	def flann_matcher(self, des1, des2):
		# FLANN parameters
		FLANN_INDEX_KDTREE = 0
		index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
		search_params = dict(checks=50)   # or pass empty dictionary

		flann = cv2.FlannBasedMatcher(index_params,search_params)
		matches = flann.knnMatch(des1,des2,k=2)

		goodMatches = []
		#print 'matches: {}'.format(len(matches))

		''' Filter out matches based on the maximum distance parameter '''
		for m in matches:
			if len(m) == 2:
				if m[0].distance < (m[1].distance*MAX_DISTANCE):
					goodMatches.append(m[0])
		if len(goodMatches) < MIN_MATCH_COUNT:
			return False
		#print "Good matches: {}".format(len(goodMatches))
		return True

		# Performs image sharpening by first subtracting
		# GaussainBlur from the original image.
	def sharpen(self, img):
		tmp = cv2.GaussianBlur(img, (15,15), 5)
		return cv2.addWeighted(img, 1.5, tmp, -0.5, 0)

# This class takes a source image file and breaks it up
# into individual cards. It seems to duplicate cards
# because the Convex hull seems to be wrapping the cards twice.
class Training():

	def __init__(self):
		self.folder = train_folder
		# Source image file which we use to build training files
		self.filename = './images/cards_black_background.png'

	def train(self):
		im = cv2.imread(self.filename)
		possible_cards = self.tracking.detect_card(im, im, 1500)
		count = 0

		# Delete existing training files
		self.del_folder()

		# Write detected cards from the source image to disk.
		for c in possible_cards:
			filename = 'trainedcard_{}'.format(count)
			self.write_image(filename, c)
			count += 1

	# Delete folter method
	def del_folder(self):
		filelist = [f for f in os.listdir(self.folder) if f.endswith('.png')]
		for f in filelist:
			os.remove(self.folder + f)

	def write_image(self, name, img):
		filename = self.folder + '/' + name + '.png'
		print 'Writing {}'.format(filename)
		cv2.imwrite(filename, img)

	# Processed image via threshold function (Testing)
	def write_processed(self, name, img):
		filename = self.folder + '/' + name + '.png'
		print 'Writing {}'.format(filename)
		blur = cv2.GaussianBlur(img,(1,1),1000)
		flag, thresh = cv2.threshold(blur, 240, 255, cv2.THRESH_BINARY)
		cv2.imwrite(filename, thresh)

class Tracking():

	def __init__(self, stats):
		self.rioImage = None
		self.canny_thres = 50
		self.stats = stats

	def get_diff(self, grey_image, grey_base):
		return cv2.absdiff(grey_image, grey_base)

	def detect_card(self, frame, grey_image, min_perim=600, thresh=200):
		edges = cv2.Canny(grey_image, thresh, thresh)

		show_debug_img('edges', edges)

		contours, hierarchy = cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

		#print 'contours {}'.format(len(contours))
		#contours = sorted(contours, key=cv2.contourArea,reverse=True)

		hull = [cv2.convexHull(cnt, True, True) for cnt in self.filter_contours(contours)]
		#lines = self.longest_lines(hull)

		filtered = []
		possibleCards = []

		# Filter out hulls by calculating the perimeter and
		# selecting the larger ones.
		if len(hull) > 2:
			for h in hull:
				a = cv2.arcLength(h, True)
				if a > min_perim:
					#h = cv2.boundingRect(h)
					points = cv2.approxPolyDP(h,0.10*a,True)
					if len(points) == 4:
						possibleCards.append(self.get_card(frame, points))
						filtered.append(h)
						#cv2.drawContours(grey_image, points, -1, (0,255,0), 3)

		infoText = 'C - resets background subtraction. D - detects card. Cards: {}'.format(len(filtered))
		putText(grey_image, infoText, (23,23))

		if (self.stats.has_stats()):
			putText(grey_image, self.stats.get(), (23, 43))

		cv2.drawContours(grey_image, filtered, -1, (0,255,0), 3)
		cv2.imshow('detected', grey_image)
		return possibleCards

	def filter_contours(self, contours):
		edges = []
		for c in contours:
			if len(c) > 10:
				edges.append(c)
		return edges

	def get_card(self, img, corners):
		#target = [(0,0), (223,0), (223,310), (0,310)]
		corners = self.rectify(corners)

		#x1, y1, x2, y2 = corners
		#self.get_length(corners)

		target = np.array([ [0,0],[449,0],[449,449],[0,449] ],np.float32)
		transform = cv2.getPerspectiveTransform(corners, target)
		warp = cv2.warpPerspective(img,transform,(450,450))

		wk = cv2.waitKey(1)
		if wk == ord('w'):
			Training().write_image('test_match_01', warp)
		if wk == ord('d'):
			matcher.match(warp)

		cv2.imshow('current_card', warp)
		return warp

	def get_length(self, corners):
		x1, y1, x2, y2 = corners
		print x1
		length = (x2-x1)**2 + (y2-y1)**2 ** 0.5
		print length

	def rectify(self, h):
		h = h.reshape((4,2))
		hnew = np.zeros((4,2),dtype = np.float32)

		add = h.sum(1)
		hnew[0] = h[np.argmin(add)]
		hnew[2] = h[np.argmax(add)]

		diff = np.diff(h,axis = 1)
		hnew[1] = h[np.argmin(diff)]
		hnew[3] = h[np.argmax(diff)]

		return hnew

# Class used to auto scan cards. Turned off at the moment.
class Scan():

	def __init__(self, camera):
		self.camera = camera

	#calculates the sum of squared differences (SSD) between two arrays
	def ssd(self, arr1, arr2):
		return np.sum((arr1-arr2)**2)

	def watch_for_card(self):
		has_moved = False
		been_to_base = False
		camera = self.camera

		global captures
		global font
		captures = []

		ret, img = camera.read()
		height, width, depth = img.shape
		n_pixels = width*height

		base = np.copy(img)
		base = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
		recent_frames = [np.copy(base)]
		#cv.ShowImage('card', base)

		while True:
			ret, img = camera.read()
			grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

			biggest_diff = max(self.ssd(grey, frame) / n_pixels for frame in recent_frames)

			#display the cam view
			cv2.putText(img,"Diff: %s" % biggest_diff, (23,23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
			cv2.imshow('win',img)
			recent_frames.append(np.copy(grey))
			if len(recent_frames) > 4:
				del recent_frames[0]

			#check for keystroke
			c = cv2.waitKey(1)
			#if there was a keystroke, reset the last capture
			if c == 27:
				return captures
			elif c == 32:
				has_moved = True
				been_to_base = True
			elif c == ord('c'):
				print 'Resetting base...'
				base = np.copy(grey)

			#if we're stable-ish
			if biggest_diff < 10:
				base_corr = min(cv2.matchTemplate(base, frame, cv2.TM_CCOEFF_NORMED) for frame in recent_frames)
				print "stable. corr = %s. moved = %s. been_to_base = %s" % (base_corr, has_moved, been_to_base)
				if base_corr > 0.01 and not been_to_base:
					has_moved = False
					been_to_base = True
					print "STATE: been to base. waiting for move"
				elif has_moved and been_to_base:
					diff = tracking.get_diff(grey, base)
					tracking.detect_card(frame, diff)
					has_moved = False
					been_to_base = False
					print "STATE: detected. waiting for go to base"
			else:
				if not has_moved:
					print "STATE: has moved. waiting for stable"
				has_moved = True

stats = Stats()
tracking = Tracking(stats)
training = Training()
matcher = Matching(stats)
timer = Timer()

testMode = True
trainMode = False
matchTest = False

def getImg(frame):
	timer.print_rate()
	return frame

def test_mode():
	ret, base = cap.read()

	while(True):
		c = cv2.waitKey(20)
		if c == ord('c'):
			stats.clear()
			ret, base = cap.read()
		if c == ord('e'):
			sys.exit
		ret, frame = cap.read()

		diff = tracking.get_diff(frame, base)
		tracking.detect_card(frame, diff)

def train():
	training.train()

	cv2.waitKey()

if __name__ == '__main__':
	cv2.namedWindow('frame')

	# Capture from built in camera.
	cap = cv2.VideoCapture(0)
	scan = Scan(cap)

	# Half resolution
	cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH,1280)
	cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT,720)

	for x in xrange(0, 5):
		cap.read()

	if trainMode:
		train()
		sys.exit

	matcher.detect_points()

	if matchTest:
		matcher = Matching()
		matcher.test()
		k = cv2.waitKey()

	if testMode:
		print 'Tracking mode'
		test_mode()
		sys.exit

	while(True):

		ret, frame = cap.read()
		#cv2.putText(frame,"Overlay test", (23,23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
		cv2.imshow('frame', frame)
		scan.watch_for_card()

		k = cv2.waitKey(1)
		if k==27: # Esc key to stop
			break

	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()

