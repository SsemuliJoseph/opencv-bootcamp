#!/usr/bin/env python
# coding: utf-8

# ## Real-Time Video Filters with OpenCV
# This tutorial demonstrates how to build an interactive real-time video filter application using OpenCV, allowing users to toggle between a live preview, a blur filter, Canny edge detection, and a corner feature detector using keyboard shortcuts.
# 
# 1. Imports and Constants Setup

# In[2]:


import cv2
import sys
import numpy

# Define mode constants
PREVIEW = 0  # Preview Mode
BLUR = 1  # Blurring Filter
FEATURES = 2  # Corner Feature Detector
CANNY = 3  # Canny Edge Detector

# Parameters for Shi-Tomasi corner detection
feature_params = dict(
    maxCorners=100, qualityLevel=0.01, minDistance=10, blockSize=3
)


# 2. Initialization and Video Source Configuration

# In[3]:


s = 0
if len(sys.argv) > 1:
  s = sys.argv[1]

image_filter = PREVIEW
alive = True
win_name = "Camera Filters"

cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
result = None
source = cv2.VideoCapture(s)


# 3. Main Processing and Interactive Loop

# In[4]:


while alive:
  has_frame, frame = source.read()
  if not has_frame:
    break

  # Flip the frame horizontally for a mirror effect
  frame = cv2.flip(frame, 1)

  if image_filter == PREVIEW:
    result = frame
  elif image_filter == CANNY:
    result = cv2.Canny(frame, 80, 150)
  elif image_filter == BLUR:
    result = cv2.blur(frame, (13, 13))
  elif image_filter == FEATURES:
    result = frame.copy()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(frame_gray, **feature_params)
    if corners is not None:
      for x, y in numpy.float32(corners).reshape(-1, 2):
        cv2.circle(result, (int(x), int(y)), 3, (0, 255, 0), -1)

  cv2.imshow(win_name, result)

  # Keyboard controls for switching filters or quitting
  key = cv2.waitKey(1)
  if key == ord("Q") or key == ord("q") or key == 27:
    alive = False
  elif key == ord("C") or key == ord("c"):
    image_filter = CANNY
  elif key == ord("B") or key == ord("b"):
    image_filter = BLUR
  elif key == ord("F") or key == ord("f"):
    image_filter = FEATURES
  elif key == ord("P") or key == ord("p"):
    image_filter = PREVIEW


# 4. Cleanup and Resource Release

# In[5]:


source.release()
cv2.destroyWindow(win_name)


# In[ ]:




