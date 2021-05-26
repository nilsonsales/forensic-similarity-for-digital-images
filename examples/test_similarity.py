'''
    Test Forensic Similarity
'''

#%%
import sys

from numpy.core.numeric import outer
sys.path.append('..')
import matplotlib.pyplot as plt
import numpy as np

import forensic_similarity as forsim #forensic similarity tool
from utils.blockimage import tile_image #function to tile image into blocks

import cv2 as cv


# images downloaded from https://www.reddit.com/user/rombouts
img = cv.imread('images/edited_2.jpg', 1)

# Define patch size depending on the image size
if img.shape[0] >= 2160:
    patch_size = 256
else:
    patch_size = 128

overlap = patch_size//2

# reference point
refPt = []

# function to display the coordinates of
# of the points clicked on the image 
def click_event(event, x, y, flags, params):
    # check for left mouse clicks
    if event == cv.EVENT_LBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)
        refPt.append((x, y))


while 0xFF & 0xFF & cv.waitKey(1) != ord('q'):
    cv.imshow("Select tile", img)

    # setting mouse hadler for the image
    # and calling the click_event() function
    cv.setMouseCallback('Select tile', click_event)

#cv.destroyAllWindows()
#cv.waitKey(1)

x, y = (refPt[-1][0]-patch_size//2, refPt[-1][1]-patch_size//2)
w, h = (patch_size, patch_size)

rect = (x,y,w,h)

img_copy = img.copy()  # create a copy to keep our original img

while 0xFF & cv.waitKey(1) != ord('q'):
    cv.rectangle(img_copy, (x,y), (x+w,y+h), (0, 255, 0), 2)
    cv.imshow('img_copy', img_copy)
cv.destroyAllWindows()
cv.waitKey(1)

#
# Save the selected tile
my_tile = img[y:y+h,x:x+w]
#plt.imshow(cv.cvtColor(my_tile, cv.COLOR_BGR2RGB))


# Cut the original image in tiles
T,xy = tile_image(img,width=patch_size,height=patch_size, 
                x_overlap=overlap,y_overlap=overlap)


# Replicate our tile to match the img size
my_tile = np.expand_dims(my_tile, axis=0)  # create an extra dimension

# Repeat the tile in a matrix
my_tile_matrix = np.tile(my_tile, (T.shape[0], 1, 1, 1) )


# ---- CALCULATE SIMILARITY ----
# Load pretrained weights
#f_weights = '../pretrained/cam_128x128/-30'
f_weights = '../pretrained/cam_'+str(patch_size)+'x'+str(patch_size)+'/-30'
sim = forsim.calculate_forensic_similarity(my_tile_matrix,T,f_weights,patch_size)


# Get the positions of tiles with low similarity
not_similar = sim < 0.5

positions = []
for i in range(0, T.shape[0]):
    if not_similar[i]:
        positions.append(xy[i])


# Plot red rectangle over tiles with low sim
alpha = 0.3
output = img.copy()

print("Plotting red squares...")
for xy in positions:
    overlay = output.copy()
    cv.rectangle(overlay, xy, (xy[0]+w-1, xy[1]+h-1), (0, 0, 255), -1)
    cv.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

cv.rectangle(output, (x,y), (x+w,y+h), (0, 255, 0), 2) # green rectangle

print("Done.")
while 0xFF & cv.waitKey(1) != ord('q'):
    cv.imshow('output', output)
cv.destroyAllWindows()
cv.waitKey(1)


# Save image to disk
#cv.imwrite('images/detected_1.jpg', output)
