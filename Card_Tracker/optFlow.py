import numpy as np
import cv2



# GLOBAL VARIABLE - Creates some random colors
color = np.random.randint(0,255,(100,3))

# CLASSES
class CardCorners:
	def __init__(self, rank, suit, x1, y1, x2, y2, x3, y3, x4, y4):
		self.type = [rank, suit]
		self.x1 = x1
		self.y1 = y1
		self.x2 = x2
		self.y2 = y2
		self.x3 = x3
		self.y3 = y3
		self.x4 = x4
		self.y4 = y4

class CardPosition:
	def __init__(self, rank, suit, x, y):
		self.type = [rank, suit]
		self.position = [x, y]

class OpticalFlow:
		# EXAMPLE CODE
	# Example "cards" for testing

	# FUNCTIONS
	# Print cards in terminal
	def printCards(self, cards):
		for card in cards:
			if (card.type[1] == 'S'):
				suit = u'\u2660'.encode('utf-8')
			elif (card.type[1] == 'C'):
				suit = u"\u2663".encode('utf-8')
			elif (card.type[1] == 'H'):
				suit = u'\u2665'.encode('utf-8')
			elif (card.type[1] == 'D'):
				suit = u'\u2666'.encode('utf-8')
			else:
				suit = card.type[1]

			string = "%s %s" % (card.type[0], suit)

			print "Card type: "
			print string
			print "Card position: "
			print card.position
			print

	# Prints the card rank and suit for each card in view
	def showCardsInFrame(self, frame, cardsInView):
		for card in cardsInView:
			if (card.type[0] == 11):
				rank = 'Jack'
			elif (card.type[0] == 12):
				rank = 'Queen'
			elif (card.type[0] == 13):
				rank = 'King'
			elif (card.type[0] == 1 or card.type[0] == 14):
				rank = 'Ace'
			elif (card.type[0] == 0):
				rank = 'Joker'
			else:
				rank = str(card.type[0])

			purple = (254,102,228)

			if (card.type[1] == 'S'):
				colour = purple
				suit = u'\u2660'.encode('utf-8')
			elif (card.type[1] == 'C'):
				colour = purple
				suit = u"\u2663".encode('utf-8')
			elif (card.type[1] == 'H'):
				colour = (0,0,255)
				suit = u'\u2665'.encode('utf-8')
			elif (card.type[1] == 'D'):
				colour = (0,0,255)
				suit = u'\u2666'.encode('utf-8')
			elif (card.type[1] == 'R'):
				colour = (0,0,255)
				suit = ' '
			elif (card.type[1] == 'B'):
				colour = purple
				suit = ' '
			else:
				suit = card.type[1]
				colour = purple

			font = cv2.FONT_HERSHEY_SIMPLEX
			string = "%s %s" % (rank, card.type[1])
			cv2.putText(frame, string, (card.position[0], card.position[1]), font, 1,colour,2)
			#frame.text((card.position[0], card.position[1]), unicode(string,"utf-8"), font=font, fill=colour)

	# Returns feature points and a list of cards to track
	def initTrackCards(self, old_gray, cardsToTrack):

		# params for ShiTomasi corner detection
		feature_params = dict( maxCorners = 10,
						   qualityLevel = 0.3,
						   minDistance = 7,
						   blockSize = 7 )

		# Create masks of type CV_8UC1 and same size as old_gray with ROI for all cards
		cardMasks = []
		i = 0
		for card in cardsToTrack:
			mask = np.zeros(old_gray.shape,np.uint8)
			roi_corners = np.array([[(card.x1,card.y1), (card.x3,card.y3), (card.x4,card.y4), (card.x2,card.y2)]], dtype=np.int32)
			cv2.fillPoly(mask, roi_corners, (255, 255, 255))
			cardMasks.append(mask)
			i += 1

		# Transfer rank and suit to output objects
		cardsInView = []
		for card in cardsToTrack:
			myCard = CardPosition(card.type[0], card.type[1], -1, -1)
			cardsInView.append(myCard)

		# Find features in region of interest
		cardFeatures = []
		for mask in cardMasks:
			p0 = cv2.goodFeaturesToTrack(old_gray, mask = mask, **feature_params)
			cardFeatures.append(p0)

		return cardFeatures, cardsInView
	#
	def trackCards(self, frame, old_gray, frame_gray, cardsInView, cardFeatures, showTracking = False):

		# Parameters for lucas kanade optical flow
		lk_params = dict( winSize  = (15,15),
							maxLevel = 2,
							criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

		toRemove = []
		cardNumber = 0
		for pOld in cardFeatures:
			# calculate optical flow
			pNew, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, pOld, None, **lk_params)

			# Select good points
			try:
				good_new = pNew[st==1]
				good_old = pOld[st==1]

				# draw the tracks and find average
				sumX = 0
				sumY = 0
				nPoints = 0
				for i, (new,old) in enumerate(zip(good_new,good_old)):
					x,y = new.ravel()
					x_old,y_old = old.ravel()

					sumX += x
					sumY += y
					nPoints += 1

					# Prints
					if (showTracking):
						cv2.line(frame, (x,y),(x_old,y_old), color[cardNumber].tolist(), 2)
						cv2.circle(frame,(x,y),5,color[cardNumber].tolist(),-1)

				if (nPoints > 0):
					cardsInView[cardNumber].position = [int(sumX/nPoints), int(sumY/nPoints)]
				else:
					cardsInView[cardNumber].position = [-1, -1]

				cardFeatures[cardNumber] = good_new.reshape(-1,1,2)


			except:
				print "Lost track of card."
				toRemove.append(cardNumber)

			cardNumber += 1

		# Remove cards with no feature points
		for i in reversed(toRemove):
			cardFeatures.pop(i)
			cardsInView.pop(i)


if __name__ == '__main__':
	optFlow = OpticalFlow()
	cardsToTrack = []
	cardsToTrack.append(CardCorners(8, 'H', 100, 100, 120, 200, 250,100, 300,300))
	cardsToTrack.append(CardCorners(8, 'H', 100, 100, 120, 200, 250,100, 300,300))
	cardsToTrack.append(CardCorners(12, 'D', 300, 300, 300, 380, 480,300, 480,380))
	cardsToTrack.append(CardCorners(12, 'D', 300, 300, 300, 380, 480,300, 480,380))
	cardsToTrack.append(CardCorners(11, 'K', 30, 30, 30, 38, 48,30, 48,38))
	cardsToTrack.append(CardCorners(11, 'C', 30, 30, 30, 38, 48,30, 48,38))

	# Setup video capture
	cap = cv2.VideoCapture(0)

	# Half resolution
	cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH,1280)
	cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT,720)

	# Read first frame, mirrored (for easy viewing on webcam)
	ret, old_frame = cap.read()
	#old_frame = cv2.flip(old_frame,1)

	# Make grayscale
	old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

	# Set up tracking
	cardFeatures, cardsInView = optFlow.initTrackCards(old_gray, cardsToTrack)

	while(1):

		# Read next frame, mirror it and create grayscale
		ret, frame = cap.read()
		#frame = cv2.flip(frame,1)
		frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# Call tracking function
		optFlow.trackCards(frame, old_gray, frame_gray, cardsInView, cardFeatures, showTracking = True)

		# Print cards in terminal
		# printCards(cardsInView)

		# Print cards in frame
		optFlow.showCardsInFrame(frame, cardsInView)

		# Show frame
		cv2.imshow('Main',frame)

		# Make current frame to old frame for the next iteration
		old_gray = frame_gray.copy()

		# Check for exit
		k = cv2.waitKey(30) & 0xff
		if k == 27:
			break

	## Ending program ##
	cv2.destroyAllWindows()
	cap.release()
