#!/usr/bin/env python

''' - Card tracking and identification
'''

import numpy as np
import cv2
import time as t
import math, sys, os
import optFlow as of
import threading
import time
from itertools import chain

# Resize scale factor for performance testing
resize720p = True

# Maximum distance used by the FlannBasedMatcher
MAX_DISTANCE = 0.63
MIN_MATCH_COUNT = 40

MIN_PERIM = 600
MAX_PERIM = 1500

# Show debug windows
debug = True

testMode = True

version3 = int(cv2.__version__.split('.')[0]) == 3

# Folder where image database is stored/written to.
train_folder = './trained_images/'

class Timer:
	def __init__(self):
		self.frames = 0
		self.start = t.time()
		self.msTimer = {}
		self.msgOutput = {}
		self.showTimer = True
		self.count = 0

	def print_rate(self):
		diff = t.time() - self.start
		self.frames += 1

		# Reset counter every 1 second
		if (diff > 1):
			# Print out time and frame rate
			print "{0} {1}".format('FPS: ', (int)(self.frames / diff))
			self.start = t.time()
			self.frames = 0

	def startMSTimer(self, msg):
		self.msTimer.setdefault(msg, []).append((t.time(), None))

	def stopMSTimer(self, msg):
		start, end = self.msTimer[msg][-1]
		diff = (t.time() - start) * 1000
		self.msgOutput[msg] = "{0}: {1} ms".format(msg, round(diff, 2))
		#print "{0}: {1} ms".format(msg, round(diff, 2))
		self.msTimer[msg][-1] = (start, diff)
		self.msTimer[msg] = []

	def printAll(self):
		self.print_rate()
		for key in self.msgOutput.iterkeys():
			print self.msgOutput[key]

def putText(img, text, pos):
	cv2.putText(img,text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))

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

	def __init__(self, stats, optFlow, display):
		self.folder = train_folder
		self.descriptors = {}
		self.stats = stats
		self.detecting = False
		self.card_queue = []
		self.optFlow = optFlow
		self.display = display

	# Worker thread for card matching
	def match_worker(self):
		foundNames = []
		timer.startMSTimer('Total sift time:')
		print 'card queue: {}'.format(len(self.card_queue))
		count = 0
		for cardImg, points in self.card_queue:
			timer.startMSTimer('Per card sift {}:'.format(count))
			success, cardName = matcher.match(cardImg)
			timer.stopMSTimer('Per card sift {}:'.format(count))
			count+=1
			if success:
				# Filter same cards
				if any(cardName in f for f in foundNames):
					print '{} added already'.format(cardName)
				else:
					foundNames.append(cardName)
					self.optFlow.addCard(points, cardName)
			else:
				stats.add_found_card('Card not found')
		if len(foundNames) > 0:
			self.optFlow.init()
		self.card_queue = []
		self.detecting = False
		timer.stopMSTimer('Total sift time:')

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
		timer.startMSTimer('Database build time cards: {}'.format(len(filelist)))
		for f in filelist:
			img = cv2.imread(self.folder + f)
			self.descriptors[f] = self.sift(img, 'training')
			print '{} Descriptors: {}'.format(f, len(self.descriptors[f]))
		timer.stopMSTimer('Database build time cards: {}'.format(len(filelist)))

	def match_all_cards(self, cards):
		if self.detecting:
			return
		self.detecting = True
		print 'detecting: {}'.format(self.detecting)
		self.optFlow.clear()
		self.stats.clear()
		self.stats.add_found_card('Recognising cards, please wait...')
		self.card_queue.extend(cards)
		thread = threading.Thread(target=self.match_worker)
		thread.start()

	# Matching given image (img) against training files.
	def match(self, img):
		if len(self.descriptors) == 0:
			self.detect_points()
		des = self.sift(self.sharpen(img), 'sift1')
		#print 'Input card descriptors: {}'.format(len(des))
		success, cardName = self.match_all(des)
		return success, cardName

	# Helper for match function, iterates all training file features.
	def match_all(self, des):
		for key in self.descriptors:
			src_des = self.descriptors[key]
			if self.flann_matcher(des, src_des, key):
				msg = 'found: {}'.format(key)
				stats.add_found_card(msg)
				print msg
				return True, key.replace('.png', '')
		return False, None

	# Performs SIFT
	def sift(self, img, name):
		gray1 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		if version3:
			sift = cv2.xfeatures2d.SIFT_create(edgeThreshold=15)
		else:
			sift = cv2.SIFT(edgeThreshold=15)
		kp, des = sift.detectAndCompute(gray1,None)
		if version3:
			sift_frame = img.copy()
			cv2.drawKeypoints(sift_frame, kp, sift_frame)
		else:
			sift_frame = cv2.drawKeypoints(img, kp)
		self.display.set_sift_frame(sift_frame)
		#show_debug_img(name, key1)
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
	def flann_matcher(self, des1, des2, card_name):
		# FLANN parameters
		FLANN_INDEX_KDTREE = 0
		index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
		search_params = dict(checks=50)   # or pass empty dictionary

		flann = cv2.FlannBasedMatcher(index_params,search_params)
		matches = flann.knnMatch(des1,des2,k=2)

		#print 'Card: {} Matches: {}'.format(card_name, len(matches));
		goodMatches = []
		#print 'matches: {}'.format(len(matches))e
		timer.startMSTimer('Flann matcher')
		''' Filter out matches based on the maximum distance parameter '''
		for m in matches:
			if len(m) == 2:
				if m[0].distance < (m[1].distance*MAX_DISTANCE):
					goodMatches.append(m[0])
		if len(goodMatches) < MIN_MATCH_COUNT:
			return False
		timer.stopMSTimer('Flann matcher')
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

	def __init__(self, stats, matcher, display):
		self.rioImage = None
		self.canny_thres = 50
		self.stats = stats
		self.matcher = matcher
		self.grey_frame = None
		self.display = display

	def get_diff(self, grey_image, grey_base):
		timer.startMSTimer('Abs diff')
		diff = cv2.absdiff(grey_image, grey_base)
		timer.stopMSTimer('Abs diff')
		return diff

	def detect_card(self, frame, grey_image, min_perim=600, thresh=200):
		timer.startMSTimer('Canny filter')
		edges = cv2.Canny(grey_image, thresh, thresh)
		timer.stopMSTimer('Canny filter')
		self.grey_frame = grey_image
		self.display.set_edges_frame(edges)

		timer.startMSTimer('findContours')
		if version3:
			conImage, contours, hierarchy = cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
		else:
			contours, hierarchy = cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
		timer.stopMSTimer('findContours')
		#print 'contours {}'.format(len(contours))
		#contours = sorted(contours, key=cv2.contourArea,reverse=True)

		hull = [cv2.convexHull(cnt, True, True) for cnt in self.filter_contours(contours)]
		#lines = self.longest_lines(hull)

		filtered = []
		possibleCards = []
		timer.startMSTimer('Convex hull')
		# Filter out hulls by calculating the perimeter and
		# selecting the larger ones.
		if len(hull) > 2:
			for h in hull:
				a = cv2.arcLength(h, True)
				if a > self.display.min_perim and a < self.display.max_perim:
						points = cv2.approxPolyDP(h,0.10*a,True)
						if len(points) == 4:
							#print 'Perim: {}'.format(a)
							#if self.check_corners(h, filtered):
							filtered.append(h)
							corners = self.rectify(points)
							possibleCards.append((self.get_card(frame, corners), corners))

		infoText = 'C - resets background subtraction. D - recognises cards. Filtered convex hulls: {}'.format(len(filtered))
		putText(frame, infoText, (23,23))
		timer.stopMSTimer('Convex hull')
		if (self.stats.has_stats()):
			putText(frame, self.stats.get(), (23, 43))

		# Run detection when 'D' pressed
		if self.display.detect_card:
			self.display.detect_card = False
			if len(possibleCards) >= 1:
				self.matcher.match_all_cards(possibleCards)
			else:
				self.stats.add_found_card('No cards in frame')

		cv2.drawContours(frame, filtered, -1, (0,255,0), 3)

		return possibleCards

	def check_corners(self, h1, hulls):
		if len(hulls) <= 1:
			return True
		count = 0
		for h2 in hulls:
			match = cv2.matchShapes(h1, h2, cv2.cv.CV_CONTOURS_MATCH_I3, 0) * 1000
			#print 'hull: {} match: {}'.format(match, count)
			count += 1
			if match < 10:
				return False
		return True

	def filter_contours(self, contours):
		edges = []
		for c in contours:
			if len(c) > 10:
				edges.append(c)
		return edges

	def get_card(self, img, corners):
		target = np.array([ [0,0],[449,0],[449,449],[0,449] ],np.float32)
		timer.startMSTimer('Perspective transform per card')
		transform = cv2.getPerspectiveTransform(corners, target)
		timer.stopMSTimer('Perspective transform per card')
		timer.startMSTimer('Warp perspective per card')
		warp = cv2.warpPerspective(img,transform,(450,450))
		timer.stopMSTimer('Warp perspective per card')

		#wk = cv2.waitKey(1)
		#if wk == ord('w'):
		#	Training().write_image('test_match_01', warp)
		self.display.set_card_frame(warp)
		return warp

	def get_length(self, corners):
		x1, y1, x2, y2 = corners
		print x1
		length = (x2-x1)**2 + (y2-y1)**2 ** 0.5
		print length

	def rectify(self, shape):
		shape = shape.reshape((4,2))
		new_shape = np.zeros((4,2),dtype = np.float32)

		add = shape.sum(1)
		new_shape[0] = shape[np.argmin(add)]
		new_shape[2] = shape[np.argmax(add)]
		diff = np.diff(shape,axis = 1)
		new_shape[1] = shape[np.argmin(diff)]
		new_shape[3] = shape[np.argmax(diff)]

		return new_shape

# Class used to auto scan cards. Turned off at the moment.
class Scan():

	def __init__(self, camera):
		self.camera = camera
		#cv2.namedWindow('frame', cv2.WINDOW_OPENGL)
		#cv2.namedWindow('Background subtraction', cv2.WINDOW_OPENGL)
		#cv2.namedWindow('current_card', cv2.WINDOW_OPENGL)

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
					base = np.copy(grey)
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

# Class which wraps the optFlow.py file.
class OptFlowWrapper:

	def __init__(self):
		self.optFlow = of.OpticalFlow()
		self.cardsToTrack = []
		self.cardFeatures = []
		self.cardsInView = []
		self.prevFrame = None
		self.enabled = False

	def initTrackCards(self):
		self.cardsToTrack.append(of.CardCorners(8, 'H', 100, 100, 120, 200, 250,100, 300,300))
		self.cardFeatures, self.cardsInView = self.optFlow.initTrackCards(self.old_gray, self.cardsToTrack)
		self.enabled = True

	def setPrevFrame(self, old_gray):
		self.old_gray = old_gray

	# Clears optical flow tracking points and disables
	def clear(self):
		self.cardsToTrack = []
		self.cardFeatures = []
		self.cardsInView = []
		self.enabled = False

	def addCard(self, cardCorners, cardNameString):
		x1, y1, x2, y2, x3, y3, x4, y4 =  list(chain.from_iterable(cardCorners))
		num, suit = cardNameString.split('_')
		ofCorners = of.CardCorners(int(num), suit, x1, y1, x2, y2, x3, y3, x4, y4)
		self.cardsToTrack.append(ofCorners)

	def init(self):
		cardFeatures, cardsInView = self.optFlow.initTrackCards(self.old_gray, self.cardsToTrack)
		self.cardFeatures.extend(cardFeatures)
		self.cardsInView.extend(cardsInView)
		self.enabled = True

	def optFlowOverlay(self, frame, frame_gray):
		if self.enabled:
			# Call tracking function
			self.optFlow.trackCards(frame, self.old_gray, frame_gray, self.cardsInView, self.cardFeatures, showTracking = True)
			self.optFlow.showCardsInFrame(frame, self.cardsInView)

class Display():

	def __init__(self):
		self.frame = None
		self.edges_frame = None
		self.card_frame = None
		self.background_frame = None
		self.sift_frame = None
		self.new_background = False
		self.detect_card = False
		self.quit = False
		self.key = -1
		self.min_perim = MIN_PERIM
		self.max_perim = MAX_PERIM
		self.timer = threading.Timer(10, self.on_identification)

	# Sets main display frame
	def set_frame(self, frame):
		#timer.print_rate()
		self.frame = frame

	def set_edges_frame(self, frame):
		self.edges_frame = frame

	def set_card_frame(self, frame):
		self.card_frame = frame

	def set_background_frame(self, frame):
		self.background_frame = frame

	def set_sift_frame(self, frame):
		self.sift_frame = frame

	def on_identification(self):
		print 'Trigger timer...'
		self.detect_card = True

	def on_min_perim_change(self, value):
		stats.clear()
		msg = 'Min perimeter: {}'.format(value)
		stats.add_found_card(msg)
		print msg
		self.min_perim = value

	def on_max_perim_change(self, value):
		stats.clear()
		msg = 'Max perimeter: {}'.format(value)
		stats.add_found_card(msg)
		print msg
		self.max_perim = value

	def run(self):
		self.setup_debug_windows()
		self.create_window('Main window')
		self.create_window('Detected card')
		cv2.createTrackbar('Min perimeter', 'Main window', MIN_PERIM, MAX_PERIM, self.on_min_perim_change)
		cv2.createTrackbar('Max perimeter', 'Main window', MAX_PERIM, MAX_PERIM, self.on_max_perim_change)

		while (True):
			self.show_if_ready('Main window', self.frame)
			self.show_if_ready('Detected card', self.card_frame)
			self.show_debug_frames()

			# Wait for keyboard press
			self.key = cv2.waitKey(30)
			if self.key != -1:
				print 'Key: {}'.format(self.key)
				self.quit = (self.key == ord('e') or self.key == 27)
				self.detect_card = (self.key == ord('d'))
				self.new_background = self.key == ord('c')
				if self.key == ord('p'):
					timer.printAll()
				print self.detect_card
			if self.quit:
				sys.exit()

	def create_window(self, name):
		cv2.namedWindow(name, cv2.WINDOW_OPENGL)

	def setup_debug_windows(self):
		if debug:
			self.create_window('Edges frame')
			self.create_window('Background frame')
			self.create_window('Sift frame')

	def show_debug_frames(self):
		if debug:
			self.show_if_ready('Edges frame', self.edges_frame)
			self.show_if_ready('Background frame', self.background_frame)
			self.show_if_ready('Sift frame', self.sift_frame)

	def show_if_ready(self, window, frame):
		if frame is not None:
			cv2.imshow(window, frame)

	@property
	def min_perim(self):
		return self.min_perim

	@property
	def max_perim(self):
		return self.max_perim

	@property
	def quit(self):
		return self.quit

	@property
	def detect_card(self):
		return self.detect_card

	@detect_card.setter
	def detect_card(self, value):
		self.detect_card = value

	@property
	def new_background(self):
		return self.new_background

	@new_background.setter
	def new_background(self, value):
		self.new_background = value

display = Display()
optFlow = OptFlowWrapper()
stats = Stats()
training = Training()
matcher = Matching(stats, optFlow, display)
tracking = Tracking(stats, matcher, display)
timer = Timer()

trainMode = False
matchTest = False

def getImg(frame):
	timer.print_rate()
	return frame

def test_mode():
	ret, base = cap.read()

	old_gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
	optFlow.setPrevFrame(old_gray)

	while(True):
		if display.quit:
			break

		if display.new_background:
			stats.clear()
			optFlow.clear()
			ret, base = cap.read()
			display.new_background = False

		timer.startMSTimer('Frame from camera')
		ret, frame = cap.read()
		timer.startMSTimer('Frame from camera')
		diff = tracking.get_diff(frame, base)
		display.set_background_frame(diff)
		timer.startMSTimer('Total edge tracking')
		tracking.detect_card(frame, diff)
		timer.stopMSTimer('Total edge tracking')
		timer.startMSTimer('OpticalFlow')
		optFlow.setPrevFrame(old_gray)
		frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		optFlow.optFlowOverlay(frame, frame_gray)
		timer.stopMSTimer('OpticalFlow')
		display.set_frame(frame)
		old_gray = frame_gray

def train():
	training.train()

	#cv2.waitKey()

if __name__ == '__main__':
	# Capture from built in camera.
	cap = cv2.VideoCapture(0)
	scan = Scan(cap)

	# Half resolution
	if resize720p:
		if version3:
			print 'Version 3'
			cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
			cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
		else:
			cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH,1280)
			cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT,720)

	for x in xrange(0, 5):
		cap.read()

	if trainMode:
		train()
		sys.exit()

	#matcher.detect_points()

	if matchTest:
		matcher = Matching()
		matcher.test()
		k = cv2.waitKey()

	if testMode:
		print 'Tracking mode'
		main_thread = threading.Thread(target=test_mode)
		main_thread.start()
		display.run()
		sys.exit()

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

