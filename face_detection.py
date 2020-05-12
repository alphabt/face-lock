# %%
import numpy as np
from PIL import Image
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
import cv2
import json
import os

raw_path = "dataset/raw"
image_filename = "keyhole1.jpg"
image_path = os.path.join(raw_path, image_filename)
img = cv2.imread(image_path)
detector = MTCNN()
result = detector.detect_faces(img)
print(result)
result = result[0]

box = result['box']
box_x, box_y, box_w, box_h = box
keypoints = result['keypoints']
left_eye = keypoints['left_eye']
right_eye = keypoints['right_eye']
nose = keypoints['nose']
mouth_left = keypoints['mouth_left']
mouth_right = keypoints['mouth_right']

print(result)

im = np.array(Image.open(image_path), dtype=np.uint8)

# Create figure and axes
fig, ax = plt.subplots(1)

# Display the image
ax.imshow(im)

# Create a Rectangle patch
# Add the patch to the Axes
rect = patches.Rectangle((box_x, box_y), box_w, box_h, linewidth=1,
                         edgecolor='r', facecolor='none')
ax.add_patch(rect)

rect = patches.Rectangle(right_eye, 10, 10, linewidth=1,
                         edgecolor='r', facecolor='r')
ax.add_patch(rect)

rect = patches.Rectangle(left_eye, 10, 10, linewidth=1,
                         edgecolor='r', facecolor='r')
ax.add_patch(rect)

rect = patches.Rectangle(nose, 10, 10, linewidth=1,
                         edgecolor='r', facecolor='r')
ax.add_patch(rect)

rect = patches.Rectangle(mouth_left, 10, 10, linewidth=1,
                         edgecolor='r', facecolor='r')
ax.add_patch(rect)

rect = patches.Rectangle(mouth_right, 10, 10, linewidth=1,
                         edgecolor='r', facecolor='r')
ax.add_patch(rect)

plt.show()

# %%


def center_crop(img, new_width=None, new_height=None):
    width = img.shape[1]
    height = img.shape[0]

    if new_width is None:
        new_width = min(width, height)

    if new_height is None:
        new_height = min(width, height)

    left = int(np.ceil((width - new_width) / 2))
    right = width - int(np.floor((width - new_width) / 2))

    top = int(np.ceil((height - new_height) / 2))
    bottom = height - int(np.floor((height - new_height) / 2))

    if len(img.shape) == 2:
        center_cropped_img = img[top:bottom, left:right]
    else:
        center_cropped_img = img[top:bottom, left:right, ...]

    return center_cropped_img


# %%
x, y, w, h = box
cropped = img[y:y+h, x:x+w]
squared = center_crop(cropped)
aligned = cv2.resize(squared, (160, 160))

# Create figure and axes
fig, ax = plt.subplots(1)

# Display the image
ax.imshow(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB))
plt.show()

# Write to disk
aligned_path = "dataset/aligned"
output_path = os.path.join(aligned_path, image_filename)
cv2.imwrite(output_path, aligned)
