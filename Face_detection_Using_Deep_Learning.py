import os
import sys
from urllib.request import urlretrieve
from zipfile import ZipFile
import cv2

# ====================================================================
# Asset Management: Download and extract model files if missing
# ====================================================================


def download_and_unzip(url, save_path):
  print(f"Downloading and extracting assets....", end="")

  # Download zip file using urllib package
  urlretrieve(url, save_path)

  try:
    # Extract zip file contents into the same directory
    with ZipFile(save_path) as z:
      z.extractall(os.path.split(save_path)[0])
    print("Done")
  except Exception as e:
    print("\nInvalid file.", e)


# Define asset URL and local destination path
URL = r"https://www.dropbox.com/s/efitgt363ada95a/opencv_bootcamp_assets_12.zip?dl=1"
asset_zip_path = os.path.join(os.getcwd(), f"opencv_bootcamp_assets_12.zip")

# Automatically download assets if they don't already exist locally
if not os.path.exists(asset_zip_path):
  download_and_unzip(URL, asset_zip_path)

# ====================================================================
# Video Stream Setup
# ====================================================================

# Default to webcam (index 0), or use command-line argument if provided
s = 0
if len(sys.argv) > 1:
  s = sys.argv[1]

source = cv2.VideoCapture(s)
win_name = "Camera Preview"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

# ====================================================================
# Deep Learning Model Initialization (Caffe SSD Face Detector)
# ====================================================================

# Load pre-trained Caffe model architecture and weights
net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel"
)

# Set model configuration parameters
in_width = 300
in_height = 300
mean = [104, 117, 123]  # Mean subtraction values for RGB channels
conf_threshold = 0.7  # Minimum confidence score to filter weak detections

# ====================================================================
# Real-Time Processing Loop
# ====================================================================

# Run continuously until the user presses the ESC key (ASCII 27)
while cv2.waitKey(1) != 27:
  has_frame, frame = source.read()
  if not has_frame:
    break

  # Mirror the frame horizontally for a natural webcam preview feel
  frame = cv2.flip(frame, 1)
  frame_height = frame.shape[0]
  frame_width = frame.shape[1]

  # Preprocess frame into a 4D blob suitable for DNN input
  blob = cv2.dnn.blobFromImage(
      frame, 1.0, (in_width, in_height), mean, swapRB=False, crop=False
  )

  # Pass blob through the network to perform inference
  net.setInput(blob)
  detections = net.forward()

  # Loop over all detections found by the model
  for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]

    # Filter out low-confidence detections below the threshold
    if confidence > conf_threshold:
      # Extract normalized bounding box coordinates and scale them to frame size
      x_top_left = int(detections[0, 0, i, 3] * frame_width)
      y_top_left = int(detections[0, 0, i, 4] * frame_height)
      x_bottom_right = int(detections[0, 0, i, 5] * frame_width)
      y_bottom_right = int(detections[0, 0, i, 6] * frame_height)

      # Draw a green bounding box around the detected face
      cv2.rectangle(
          frame,
          (x_top_left, y_top_left),
          (x_bottom_right, y_bottom_right),
          (0, 255, 0),
      )

      # Format the confidence label string
      label = "Confidence: %.4f" % confidence
      label_size, base_line = cv2.getTextSize(
          label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
      )

      # Draw a filled background rectangle for text readability
      cv2.rectangle(
          frame,
          (x_top_left, y_top_left - label_size[1]),
          (x_top_left + label_size[0], y_top_left + base_line),
          (255, 255, 255),
          cv2.FILLED,
      )
      # Overlay the confidence text onto the frame
      cv2.putText(
          frame,
          label,
          (x_top_left, y_top_left),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.5,
          (0, 0, 0),
      )

  # Calculate and display the model inference performance time (in ms)
  t, _ = net.getPerfProfile()
  label = "Inference time: %.2f ms" % (t * 1000.0 / cv2.getTickFrequency())
  cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

  # Display the annotated frame in the GUI window
  cv2.imshow(win_name, frame)

  

# Clean up resources upon exit
source.release()
cv2.destroyWindow(win_name)