import numpy as np
import cv2
import time as t


# Test frame by frame video capture from internal camera
# on Macbook Pro.
cap = cv2.VideoCapture(0)

frames = 0
start = t.time()

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',gray)

    diff = t.time() - start
    frames += 1

    # Print out time and frame rate
    print "{0} {1}".format(diff, (frames / diff))

    # Reset counter every 1 second
    if (diff > 1):
      start = t.time()
      frames = 0

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
