#!/usr/bin/env python

'''
multi_tracker.py [optional video file]

If no video file is provided, internal camera is selected. Use
mouse to select areas for tracking targets.

'p' - toggles pause
'c' - clear tracking targets
'esc' - quit program

Tested with OpenCV version 2.4.11

1. Program peforms ORB detection and extraction for the target frame, using the
    ORB detectAndCompute method and filters for the region specified by the user. (Oriented FAST and Rotated BRIEF)

2. For every subsequent frame, compute detection and extraction of features.

3. Perform Fast Approximate Nearest Neighbor Search Library (FLANN) to match key points.

4. Performs homography calculation with RANSAC

5. Performs perspective transform

5. Visualises on the frame.

'''

import numpy as np
import cv2, sys
import time as t

# Default FLANN parameters taken from OpenCV examples
FLANN_INDEX_KDTREE = 1
FLANN_INDEX_LSH    = 6
FLANN_PARAMETERS = dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6,
                   key_size = 12,
                   multi_probe_level = 1)

# Minimum matches required by the flann feature detector before we attempt the homography matrix and perspective transform operations.
MIN_MATCH_COUNT = 6

# Max features
ORB_FEATURES = 1000

# Maximum distance used by the FlannBasedMatcher
MAX_DISTANCE = 0.5

#Allowed reprojection error using by RANSAC
RANSAC_THRESHOLD = 3.0

showPoints = False

visualisePoints = True

''' Timer class so we can caculate some tracking performance metrics '''
class Timer:
    def __init__(self):
        self.frames = 0
        self.start = t.time()
        self.msTimer = {}
        self.showTimer = True
        self.count = 0

    @property
    def showTimer(self):
        return self.showTimer

    def printFPSRate(self):
        diff = t.time() - self.start
        self.frames += 1
        # Print out time and frame rate
        #if self.showTimer:
        print "{0} {1}".format('FPS: ', (int)(self.frames / diff))
        self.start = t.time()
        self.frames = 0

        # Reset counter every 1 second and print
        #if (np.int16(diff) == 1):

    def startMSTimer(self, msg):
        self.msTimer.setdefault(msg, []).append((t.time(), None))

    def stopMSTimer(self, msg):
        start, end = self.msTimer[msg][-1]
        diff = (t.time() - start) * 1000
        print "{0}: {1} ms".format(msg, round(diff, 2))
        self.msTimer[msg][-1] = (start, diff)
        self.msTimer[msg] = []

''' Instantiate timer class '''
timer = Timer()

''' Class stores points and descriptors for the tracking target '''
class Target:

    def __init__(self):
        # Initalise empty arrays for points and descriptors.
        self.points = []
        self.descriptors = []

    def addPoints(self, points_descriptors, rectangle):
        x0, y0, x1, y1 = rectangle
        self.rectangle = rectangle
        for point, descriptor in points_descriptors:
            # Check if detected points fit within the selected
            # box/rectangle and add to points array.
            if x0 <= point.pt[0] <= x1 and y0 <= point.pt[1] <= y1:
                self.points.append(point)
                self.descriptors.append(descriptor)

    @property
    def points(self):
        return self.points

    @property
    def descriptors(self):
        return self.descriptors

    @property
    def rectangle(self):
        return self.rectangle

''' Tracking result performs homography matrix and perspective transform operations '''
class TrackResult:

    def __init__(self, target, src_points, dst_points):
        self.target = target
        src, dst = np.float32((src_points, dst_points))
        self.src_points = src
        self.dst_points = dst
        self.valid = False
        self.findHomographyAndTransform()

    ''' Finds the homography matrix between target and input points. We then perform a perspective transform so we can visualise the tracked object(s) '''
    def findHomographyAndTransform(self):
        timer.startMSTimer("Find homography RANSAC")
        H, status = cv2.findHomography(self.src_points, self.dst_points, cv2.RANSAC, RANSAC_THRESHOLD)
        timer.stopMSTimer("Find homography RANSAC")
        if (status is None):
            return None
        status = status.ravel() != 0
        if status.sum() < MIN_MATCH_COUNT:
            return None

        timer.startMSTimer("Perspective transform")
        x0, y0, x1, y1 = self.target.rectangle
        rectangle = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        transform = cv2.perspectiveTransform(rectangle.reshape(1, -1, 2), H).reshape(-1, 2)
        timer.stopMSTimer("Perspective transform")

        self.rectangle = transform
        self.src_points = self.src_points[status]
        self.dst_points = self.dst_points[status]
        self.valid = True

    @property
    def rectangle(self):
        return self.rectangle

    @property
    def valid(self):
        return self.valid

    @property
    def srcPoints(self):
        return self.src_points

    @property
    def dstPoints(self):
        return self.dst_points

''' This class encapsulates the ORB feature detector
    and FlannBasedMatcher and stores target points to search
    for in the targets array. '''
class ORBTracker:
    def __init__(self):
        self.detector = cv2.ORB( nfeatures = ORB_FEATURES, edgeThreshold = 2, patchSize = 31, WTA_K=3, scoreType=1)
        self.matcher = cv2.FlannBasedMatcher(FLANN_PARAMETERS,
            dict(checks = 50))
        self.targets = []

    ''' Step 1: Detects features for the given target boundary '''
    def addTarget(self, image, rectangle, visualiseFrame):
        x0, y0, x1, y1 = rectangle
        points, descriptors = self.detectFeatures(image)

        #self.visualisePoints(points, visualiseFrame)

        # Instantiate the target class
        target = Target()

        # add points and descriptors to the target class.
        target.addPoints(zip(points, descriptors), rectangle)

        # Add descriptors to the flann matcher.
        descriptors = np.uint8(target.descriptors)

        print "Descriptors found: {}".format(len(descriptors))
        self.matcher.add([descriptors])

        # Store target in the targets array.
        self.targets.append(target)

    def visualisePoints(self, visualiseFrame):
        if len(self.targets) > 0:
            for target in self.targets:
                if showPoints:
                    print "Number of points: {}".format(len(target.points))
                for point in target.points:
                    x, y = np.int32(point.pt)
                    #print "X: {} Y: {}".format(x, y)
                    cv2.circle(visualiseFrame, (x, y), 1, (200, 255, 200))

    def visualisePoints2(self, visualiseFrame, points):
        for point in points:
            x, y = point
            if showPoints:
                print "X: {} Y: {}".format(x, y)
            cv2.circle(visualiseFrame, (x, y), 1, (200, 255, 200))


    ''' Performs tracking on provided frame '''
    def track(self, frame):
        points, descriptors = self.detectFeatures(frame)
        if len(points) < MIN_MATCH_COUNT:
            return []

        ''' Use FLANN matcher to find good matches. '''
        knnMatches = self.getKnnMatches(descriptors)
        if knnMatches is None:
            return []

        # Array to store tracking results
        trackResults= []

        # We can have more than 1 tracking area, so we need to process each of them.
        for key, value in knnMatches.iteritems():
            index = key
            matches = value
            if len(matches) < MIN_MATCH_COUNT:
                continue
            target = self.targets[index]
            target_points = target.points
            src, dst = self.getTargetInputPoints(matches, target_points, points)
            trackResult = TrackResult(target, src, dst)
            if trackResult.valid:
                trackResults.append(trackResult)

        return trackResults

    ''' This method initiates the FlannBasedMatcher with given
        set of descriptors. Results are then filtered based on
        the minimum distance from the descriptors stored in the matcher (added in the addTarget method. '''
    def getKnnMatches(self, descriptors):
        timer.startMSTimer("FlannBasedMatcher")
        knnMatches = self.matcher.knnMatch(descriptors, k = 2)
        timer.stopMSTimer("FlannBasedMatcher")
        goodMatches = []
        retDict = {}

        ''' Filter out matches based on the maximum distance parameter '''
        for m in knnMatches:
            # Ensure tuple has two values.
            if len(m) == 2:
                if m[0].distance < (m[1].distance*MAX_DISTANCE):
                    goodMatches.append(m[0])
        if len(goodMatches) < MIN_MATCH_COUNT:
            return None
        print "Good matches: {}".format(len(goodMatches))
        for match in goodMatches:
            if match.imgIdx not in retDict:
                retDict[match.imgIdx] = []
            newList = retDict[match.imgIdx]
            newList.append(match)
        return retDict

    ''' Gets the points in the target image and points in the input frame. This is then used with the homography matrix. '''
    def getTargetInputPoints(self, matches, target_points, input_points):
        # Initalise arrays to store points
        src, dst = [], []
        for match in matches:
            target_point = target_points[match.trainIdx].pt
            input_point = input_points[match.queryIdx].pt
            src.append(target_point)
            dst.append(input_point)

        #print "Target points: {}".format(src)
        #print "Input points: {}".format(dst)
        return src, dst

    ''' Uses ORB matcher to detect keypoints and features for given
        frame '''
    def detectFeatures(self, frame):
        timer.startMSTimer("Detect keypoints")
        keypoints = self.detector.detect(frame, None)
        timer.stopMSTimer("Detect keypoints")
        timer.startMSTimer("Compute descriptors")
        keypoints, descriptors = self.detector.compute(frame, keypoints)
        timer.stopMSTimer("Compute descriptors")
        if descriptors is None:
            descriptors = []
        return (keypoints, descriptors)

    def clear(self):
        print "Clearing tracking targets"
        self.targets = []
        self.matcher.clear()

''' This class provides user mouse input for
    selecting the tracking area '''
class MouseFrame:

    def __init__(self, namedWindow, callback):
        self.callback = callback
        self.start = None
        self.rectangle = None
        self.drawing = False
        cv2.setMouseCallback(namedWindow, self.mouseEvents)

    ''' Mouse callback event handler '''
    def mouseEvents(self, event, x, y, flags, param):
        x, y = np.int32([x, y])
        # Mouse down event
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start = (x, y)
            self.drawing = True
        # Mouse up event
        if event == cv2.EVENT_LBUTTONUP:
            print "Mouse up"
            self.drawing = False
            if self.rectangle:
                self.callback(self.rectangle)
            self.start = None
            self.rectangle = None
        # Mouse move event
        if event == cv2.EVENT_MOUSEMOVE and self.drawing == True:
            xo, yo = self.start
            x0, y0 = np.minimum([xo, yo], [x, y])
            x1, y1 = np.maximum([xo, yo], [x, y])
            self.rectangle = None
            if x1-x0 > 0 and y1-y0 > 0:
                self.rectangle = (x0, y0, x1, y1)

    def draw(self, vis):
        if not self.rectangle:
            return False
        x0, y0, x1, y1 = self.rectangle
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
        return True

    @property
    def drawing(self):
        return self.drawing

class App:
    def __init__(self, src):
        self.video = cv2.VideoCapture(src)
        self.frame = None
        self.paused = False
        self.visualise = None

        # Initalise the tracker class
        self.tracker = ORBTracker()
        cv2.namedWindow('Video')
        self.mouseFrame = MouseFrame('Video', self.mouseCallback)

    def mouseCallback(self, rectangle):
        print "Adding image target: {}".format(rectangle)
        self.tracker.addTarget(self.frame, rectangle, self.visualise)

    ''' Draws the rectangle overlay on the frame '''
    def drawOverlay(self, frame, tracks):
        for track in tracks:
            cv2.polylines(frame, [np.int32(track.rectangle)], True, (0, 255, 0), 2)
            self.tracker.visualisePoints2(frame, track.dstPoints)

    def close(self):
        self.tracker.clear()
        self.video.release()
        cv2.destroyAllWindows()

    def start(self):
        count = 0

        timer.showTimer = True
        while True:
            playing = not self.paused and not self.mouseFrame.drawing
            if playing or self.frame is None:
                ret, frame = self.video.read()
                if not ret:
                    break
                self.frame = frame.copy()

            self.visualise = self.frame.copy()

            # Pause on first frame of video.
            if (count == 0):
                self.paused = True
            count += 1
            # Overflow protection
            if (count == 100):
                count =1

            if playing:
                timer.startMSTimer('Total tracking time')
                tracks = self.tracker.track(self.frame)
                self.drawOverlay(self.visualise, tracks)
                timer.stopMSTimer('Total tracking time')
                timer.printFPSRate()

            self.mouseFrame.draw(self.visualise)

            if visualisePoints:
                self.tracker.visualisePoints(self.visualise)

            cv2.imshow('Video', self.visualise)

            waitKey = cv2.waitKey(1)
            if waitKey == ord('c'):
                self.tracker.clear()
            if waitKey == ord('p'):
                self.paused = not self.paused
            if waitKey == 27:
                self.close()
                break

def init():
    print __doc__
    try:
        videoSource = sys.argv[1]
    except:
        videoSource = 0
    app = App(videoSource)
    app.start()

init()
