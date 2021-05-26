'''
    Test Forensic Similarity
'''

#%%
import sys
sys.path.append('..')
import matplotlib.pyplot as plt
import numpy as np

import forensic_similarity as forsim #forensic similarity tool
from utils.blockimage import tile_image #function to tile image into blocks

#%%
#LOAD IMAGES
I0 = plt.imread('./images/0_google_pixel_1.jpg')
I1 = plt.imread('./images/1_google_pixel_1.jpg')
I2 = plt.imread('./images/2_asus_zenphone_laser.jpg')

#SHOW IMAGES
fig,ax = plt.subplots(1,3)
ax[0].imshow(I0); ax[1].imshow(I1); ax[2].imshow(I2)
ax[0].set_title('Image 0'); ax[1].set_title('Image 1'); ax[2].set_title('Image 2')
plt.show()

#%%
#Get tiles/patches from images
patch_size = 256
overlap = 128

#tiles and xy coordinates of each tile for image 0
T0,xy0 = tile_image(I0,width=patch_size,height=patch_size, 
                x_overlap=overlap,y_overlap=overlap)

#tiles and xy coordinates of each tile for image 1
T1,xy1 = tile_image(I1,width=patch_size,height=patch_size,
                x_overlap=overlap,y_overlap=overlap)

#tiles and xy coordinates of each tile for image 2
T2,xy2 = tile_image(I2,width=patch_size,height=patch_size,
                x_overlap=overlap,y_overlap=overlap)

#SHOW RANDOM TILES
ind0 = np.random.randint(0,len(T0)) #select random index
ind1 = np.random.randint(0,len(T1))
ind2 = np.random.randint(0,len(T2))

fig,ax = plt.subplots(1,3) #show randomly selected tiles
ax[0].imshow(T0[ind0])
ax[1].imshow(T1[ind1])
ax[2].imshow(T2[ind2])

ax[0].set_title('Image 0 \n x ={}, y={}'.format(xy0[ind0][0],xy0[ind0][1]))
ax[1].set_title('Image 1 \n x ={}, y={}'.format(xy1[ind1][0],xy1[ind1][1]))
ax[2].set_title('Image 2 \n x ={}, y={}'.format(xy2[ind2][0],xy2[ind2][1]))

plt.show()

#%%
#Randomly select N tiles
N = 1000

inds0 = np.random.randint(0,len(T0),size=N) #select random indices
inds1 = np.random.randint(0,len(T1),size=N) 
inds2 = np.random.randint(0,len(T2),size=N)

X0 = T0[inds0] #vector of randomly selected image tiles
X1 = T1[inds1] #vector of randomly selected image tiles
X2 = T2[inds2] #vector of randomly selected image tiles

#%%
#Calculate Forensic Similarity between images
f_weights = '../pretrained/cam_256x256/-30' #path to pretrained CNN weights
sim_0_1 = forsim.calculate_forensic_similarity(X0,X1,f_weights,patch_size) #between tiles from image 0 and image 1
sim_0_2 = forsim.calculate_forensic_similarity(X0,X2,f_weights,patch_size) #between tiles from image 0 and image 2
sim_1_2 = forsim.calculate_forensic_similarity(X1,X2,f_weights,patch_size) #between tiles from image 1 and image 2

#%%
#plot distributions
fig,ax = plt.subplots(1)
ax.hist(sim_0_1,50,label='Image0-Image1')
ax.hist(sim_0_2,50,label='Image0-Image2')
ax.hist(sim_1_2,50,label='Image1-Image2')
ax.legend()
plt.show()

#image0 and image1 are from the same camera model, and have high forensic similarity
#image0 and image2 are from different camera models, and have low forensic similarity
#image1 and image2 are from different camera models, and have low forensic similarity


#%%
# Test comparing the first tile to another image
# Select first tile
X0 = T0[0]
X0 = np.expand_dims(X0, axis=0)  # create an extra dimension

# Repeat the first tile in a matrix
X0 = np.tile(X0, (X1.shape[0], 1, 1, 1) ) 

# Lead pretrained weights
f_weights = '../pretrained/cam_256x256/-30'
sim_0_1 = forsim.calculate_forensic_similarity(X0,X1,f_weights,patch_size) #between tiles from image 0 and image 1

#%%
import cv2 as cv

patch_size = 128
overlap = patch_size//2

# image downloaded from https://www.reddit.com/user/rombouts
img = cv.imread('images/edited_1.jpg', 1)

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

x, y = (refPt[0][0]-patch_size//2, refPt[0][1]-patch_size//2)
w, h = (patch_size, patch_size)

rect = (x,y,w,h)

img_copy = img.copy()  # create a copy to keep our original img

while 0xFF & cv.waitKey(1) != ord('q'):
  cv.rectangle(img_copy, (x,y), (x+w,y+h), (0, 255, 0), 2)
  cv.imshow('img', img_copy)
cv.destroyAllWindows()
cv.waitKey(1)

#%%
# Save the selected tile
selected_tile = img[y:y+h,x:x+w]
plt.imshow(cv.cvtColor(selected_tile, cv.COLOR_BGR2RGB))


# Cut the original image in tiles
T,xy = tile_image(img,width=patch_size,height=patch_size, 
                x_overlap=overlap,y_overlap=overlap)


# Replicate our tile to match the img size
selected_tile = np.expand_dims(selected_tile, axis=0)  # create an extra dimension

# Repeat the tile in a matrix
X0 = np.tile(selected_tile, (T.shape[0], 1, 1, 1) )

# To do:
# 1. Achar o tile selecionado na lista
# 2. Comparar todos os tiles com ele
# 3. pintar de vermelho os tiles com sim < 0.5

# Draw red rectangle:
# cv.rectangle(img, (x,y), (x+w,y+h), (0, 0, 255), -1) # how to add alpha channel ?
# %%
