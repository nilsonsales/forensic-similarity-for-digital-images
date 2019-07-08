
# -*- coding: utf-8 -*-
"""
Several functions to extract image tiles

@author: Owen Mayer - MISL - om82@drexel.edu
"""

import numpy as np

#evenly space tiles from top left of image. Do not include any tiles that go over the right or bottom edge
def tile_image(I,width,height,x_overlap=0,y_overlap=0):
    jnds = range(0,I.shape[0]-height+1,height-y_overlap) #block starting y indices
    inds = range(0,I.shape[1]-width+1,width-x_overlap)   #block startin x indicies
    N = len(jnds)*len(inds)
    
    if I.ndim == 3:
        X = np.zeros((N,height,width,I.shape[2]),dtype=I.dtype) #Initialize data holder
    elif I.ndim == 2:
        X = np.zeros((N,height,width),dtype=I.dtype) #Initialize data holder
    else:
        raise ValueError('Expecting a single or multi-channel image array')   #if not 2d or 3d, tell us
        
    xy = [];
    count = 0;
    for jj in jnds:
        for ii in inds:
            X[count] = I[jj:jj+height,ii:ii+width]
            xy.append((ii,jj))
            count += 1
    return X, xy

#evenly space n_x tiles in x direction, and n_y tiles in y direction
#span image, such that they start at top and left edges, and end at right and bottom edges
#optionally, snap tiles to nearest "snap_to" amount. Default to 16 to align to JPEG grid
#set snap_to=1 to disable grid snapping
def span_image(I,width,height,n_x=10,n_y=10,snap_to=16):
    inds = np.linspace(0,I.shape[1]-width,n_x)
    inds = np.round(inds/snap_to)*snap_to #round to nearest snap_to'th
    i_snap_max = np.floor((I.shape[1]-width)/snap_to)*snap_to
    inds = np.clip(inds,0,i_snap_max) 
    inds = inds.astype(int) #convert to int
    
    jnds = np.linspace(0,I.shape[0]-height,n_y)
    jnds = np.round(jnds/snap_to)*snap_to #round to nearest snap_to'th
    j_snap_max = np.floor((I.shape[0]-height)/snap_to)*snap_to
    jnds = np.clip(jnds,0,j_snap_max) 
    jnds = jnds.astype(int) #convert to int
    
    N = n_x*n_y
    
    if I.ndim == 3:
        X = np.zeros((N,height,width,I.shape[2]),dtype=I.dtype) #Initialize data holder
    elif I.ndim == 2:
        X = np.zeros((N,height,width),dtype=I.dtype) #Initialize data holder
    else:
        raise ValueError('Expecting a single or multi-channel image array')   #if not 2d or 3d, tell us
        
    xy = [];
    count = 0;
    for jj in jnds:
        for ii in inds:
            X[count] = I[jj:jj+height,ii:ii+width]
            xy.append((ii,jj))
            count += 1
    return X, xy

#evenly space tiles by overlap amount
#span image, such that they start at top and left edges, and end at right and bottom edges
#optionally, snap tiles to nearest "snap_to" amount. Default to 16 to align to JPEG grid
#set snap_to=1 to disable grid snapping
def span_image_by_overlap(I,width,height,x_overlap=0,y_overlap=0,snap_to=16):
    #this will evenly space tiles covering the whole image
    #overlap between tiles is the closest evenly disivible amount to x_overlap and y_overlap
    n_x = (I.shape[1]-width)/(width-x_overlap)
    n_x = int(np.round(n_x))
    
    n_y = (I.shape[0]-height)/(height-y_overlap)
    n_y = int(np.round(n_y))
    
    X, xy = span_image(I,width,height,n_x+1,n_y+1,snap_to=snap_to)
    
    return X, xy
