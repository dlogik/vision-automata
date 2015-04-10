''' - This program tests SIFT tracking using a Python implementation.
    - Clicking the image will start tracking using a square of
      pre-defined size.
    - Pressing T will toggle multi-thread mode for performance testing.
    - Pressing ESC will exit
    - Adjust "resize" and "size" variables as necessary
'''

import numpy as np
import cv2
import time as t

from multiprocessing.pool import ThreadPool
from collections import deque

# Capture from built in camera.
cap = cv2.VideoCapture(0)

# Resize scale factor for performance testing
resize = 0.7

# Size of square which we use to track.
size = 50


#----
drawing = False
track = False
ix, iy = 0,0

cv2.namedWindow('frame')

# mouse callback function
def draw_rec(event,x,y,flags,param):
  global ix,iy,drawing,mode

  if event == cv2.EVENT_LBUTTONDOWN:
    drawing = True
    ix, iy = x,y

# Setup mouse callback.
cv2.setMouseCallback('frame', draw_rec)

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

class DummyTask:
    def __init__(self, data):
        self.data = data
    def ready(self):
        return True
    def get(self):
        return self.data

class Tracking():

  def __init__(self):
    self.rioImage = None

  def SnapMouse(self, frame, x, y):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.rectangle(frame,(ix-size,iy-size),(ix+size,iy+size),(0,255,0),1)
    #rio = cv2.rectangle((ix-size,iy-size),(ix+size,iy+size),(0,255,0),1)

    self.rioImg = frame[y-size:y+size, x-size:x+size]

  def Track(self, frame, t0):
    img1 = self.rioImg          # queryImage
    img2 = frame # trainImage
    imgColour = frame

    img2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    MIN_MATCH_COUNT = 4

    # Initiate SIFT detector
    sift = cv2.SIFT()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
          good.append(m)

    if len(good)>MIN_MATCH_COUNT:
      src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
      dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
      M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
      h,w = img1.shape
      pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
      dst = cv2.perspectiveTransform(pts,M)

      cv2.polylines(imgColour,[np.int32(dst)],True,255,1, cv2.CV_AA)
      #print np.int32(dst)

      return imgColour

    else:
      print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
       # matchesMask = None
      return imgColour


tracking = Tracking()
timer = Timer()

threadn = cv2.getNumberOfCPUs()
pool = ThreadPool(processes = threadn)
pending = deque()

threaded_mode = True

def getImg(frame):
  timer.print_rate()
  return frame

while(True):
  while len(pending) > 0 and pending[0].ready():
    res = pending.popleft().get()
    timer.print_rate()
    cv2.imshow('frame', res)
  if len(pending) < threadn:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (0,0), fx=resize, fy=resize)
    if track:
      if threaded_mode:
        task = pool.apply_async(tracking.Track, (frame.copy(), 1))
      else:
        task = DummyTask(tracking.Track(frame, 1))
    else:
      task = DummyTask(getImg(frame))
    pending.append(task)

  if (drawing):
    track = False
    tracking.SnapMouse(frame, ix, iy)
    drawing = False
    track = True

  k = cv2.waitKey(33)
  if k==27:    # Esc key to stop
    break
  elif k== ord('t'):  # toggle multi thread mode.
    threaded_mode = not threaded_mode
    print 'Thread mode: {0}'.format(threaded_mode)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

