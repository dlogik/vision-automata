#!/usr/bin/env python

'''
multi_tracker.py [optional video file]

If no video file is provided, internal camera is selected. Use
mouse to areas tracking targets.

'p' - toggles pause
'c' - clear tracking targets
'esc' - quit program

'''

import numpy as np
import cv2, sys

# Default FLANN parameters taken from OpenCV
FLANN_INDEX_KDTREE = 1
FLANN_INDEX_LSH    = 6
FLANN_PARAMETERS = dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6,
                   key_size = 12,
                   multi_probe_level = 1)

MIN_MATCH_COUNT = 10
ORB_FEATURES = 1000

# Maximum distance used by the FlannBasedMatcher
MAX_DISTANCE = 0.75


''' Class stores points and descriptors for tracking target '''
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
        p0, p1 = np.float32((src_points, dst_points))
        self.src_points = p0
        self.dst_points = p1
        self.valid = False
        self.findHomographyAndTransform()

    ''' Finds the homography matrix between target and input points. We then perform a perspective transform so we can visualise the tracked object(s) '''
    def findHomographyAndTransform(self):
        H, status = cv2.findHomography(self.src_points, self.dst_points, cv2.RANSAC, 3.0)
        if (status is None):
            return None
        status = status.ravel() != 0
        if status.sum() < MIN_MATCH_COUNT:
            return None
        x0, y0, x1, y1 = self.target.rectangle
        rectangle = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        transform = cv2.perspectiveTransform(rectangle.reshape(1, -1, 2), H).reshape(-1, 2)
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
        self.detector = cv2.ORB( nfeatures = ORB_FEATURES )
        self.matcher = cv2.FlannBasedMatcher(FLANN_PARAMETERS,
            dict(checks = 50))
        self.targets = []

    def addTarget(self, image, rectangle):
        x0, y0, x1, y1 = rectangle
        points, descriptors = self.detectFeatures(image)

        # Instantiate the target class
        target = Target()

        # add points and descriptors to the target class.
        target.addPoints(zip(points, descriptors), rectangle)

        # Add descriptors to the flann matcher.
        descriptors = np.uint8(target.descriptors)
        self.matcher.add([descriptors])

        # Store target in the targets array.
        self.targets.append(target)

    ''' Performs tracking on provided frame '''
    def track(self, frame):
        points, descriptors = self.detectFeatures(frame)
        if len(points) < MIN_MATCH_COUNT:
            return []

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
            p0, p1 = self.getTargetInputPoints(matches, target_points, points)
            trackResult = TrackResult(target, p0, p1)
            if trackResult.valid:
                trackResults.append(trackResult)

        return trackResults

    ''' This method initiates the FlannBasedMatcher with given
        set of descriptors. Results are then filtered based on
        the minimum distance from the descriptors stored in the matcher (added in the addTarget method. '''
    def getKnnMatches(self, descriptors):
        knnMatches = self.matcher.knnMatch(descriptors, k = 2)
        goodMatches = []
        retDict = {}

        for m in knnMatches:
            # Ensure tuple has two values.
            if len(m) == 2:
                if m[0].distance < (m[1].distance*MAX_DISTANCE):
                    goodMatches.append(m[0])
        if len(goodMatches) < MIN_MATCH_COUNT:
            return None
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
        return src, dst

    ''' Uses ORB matcher to detect keypoints and features for given
        frame '''
    def detectFeatures(self, frame):
        keypoints, descriptors = self.detector.detectAndCompute(frame, None)
        if descriptors is None:
            descriptors = []
        return (keypoints, descriptors)

    def clear(self):
        '''Remove all targets'''
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
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start = (x, y)
            self.drawing = True
        if event == cv2.EVENT_LBUTTONUP:
            print "Mouse up"
            self.drawing = False
            if self.rectangle:
                self.callback(self.rectangle)
            self.start = None
            self.rectangle = None
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

        # Initalise the tracker class
        self.tracker = ORBTracker()
        cv2.namedWindow('Video')
        self.mouseFrame = MouseFrame('Video', self.mouseCallback)

    def mouseCallback(self, rectangle):
        print "Adding image target: {}".format(rectangle)
        self.tracker.addTarget(self.frame, rectangle)

    ''' Draws the rectangle overlay on the frame '''
    def drawOverlay(self, frame, tracks):
        for track in tracks:
            cv2.polylines(frame, [np.int32(track.rectangle)], True, (0, 255, 0), 2)

    def start(self):
        count = 0
        while True:
            playing = not self.paused and not self.mouseFrame.drawing
            if playing or self.frame is None:
                ret, frame = self.video.read()
                if not ret:
                    break
                self.frame = frame.copy()

            visualise = self.frame.copy()

            # Pause on first frame of video.
            if (count == 0):
                self.paused = True
            count += 1

            if playing:
                tracks = self.tracker.track(self.frame)
                self.drawOverlay(visualise, tracks)

            self.mouseFrame.draw(visualise)
            cv2.imshow('Video', visualise)

            waitKey = cv2.waitKey(1)
            if waitKey == ord('c'):
                self.tracker.clear()
            if waitKey == ord('p'):
                self.paused = not self.paused
            if waitKey == 27:
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
