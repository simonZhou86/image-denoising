import sys, os, math

import numpy as np

from PIL import Image

from OpenGL.GLUT import *
from OpenGL.GL import *
from OpenGL.GLU import *


def ft1D( signal ):
	
  return np.fft.fft( signal )

# Input is a 2D numpy array of complex values.
# Output is the same.

def forwardFT( image ):

  # YOUR CODE HERE
  #
  # You must replace this code with your own, keeping the same function name are parameters.
  row = image.shape[0] 
  column = image.shape[1]

  #result_image = np.empty([row, column], np.complex)

  result_image = np.array(image, dtype = "complex")

  for r in range(row): # apply forward transform, treat column as a 1D singal, compute F(x,v)
    result_image[r,:] = ft1D(result_image[r,:])
  for c in range(column): # based on F(x,v), compute F(u,v)
    result_image[:,c] = ft1D(result_image[:,c])
  return result_image


 
def inverseFT( image ):
	
  # YOUR CODE HERE
  #
  # You must replace this code with your own, keeping the same function name are parameters.
  # the inverse of fourier transform is just the conjugate of conjugate of the FT image
  result_image = np.array(np.conjugate(image)) # compute the comjugate of the image
  row = result_image.shape[0] 
  column = result_image.shape[1]

  for r in range(row):
    result_image[r,:] = ft1D(result_image[r,:]) / row # do the inverse transform on F(x,v)  
  for c in range(column):
    result_image[:,c] = ft1D(result_image[:,c]) / column # do the inverse transform  on x for f(x,y) using F(x,v)

  final_image =  np.conjugate(result_image) # conjugate back

  return final_image



def multiplyFTs( image, filter ):
	
  # YOUR CODE HERE
  image_w = image.shape[0]
  image_h = image.shape[1]
  ft_filter_shifted = filter.copy()
  filter_w = ft_filter_shifted.shape[0]
  filter_h = ft_filter_shifted.shape[1]
  #print(image.shape)

  # after simplyfy the above equation, we have e^{pi*i*x} for x-direction and e^{pi*i*y} for y-direction
  # combine x and y we have e^{pi*i*x} * e^{pi*i*y} = e^{pi*i*(x+y)} = (e^{pi*i})^(x+y) = (-1)^(x+y) by euler formula
  for i in range(0, filter_w):
    for j in range(0, filter_h):
      ft_filter_shifted[i][j] = ft_filter_shifted[i][j] * ((-1) ** (i+j))
  
  #print(ft_filter_shifted.shape)
  result_image = np.empty([image_w, image_h], np.complex) # generate a blank image

  result_image = ft_filter_shifted * image # multiply in frequency domain = convolve in spatial domain, and set it to result image

  return result_image 



def modulatePixels( image, x, y, isFT ):
  # YOUR CODE HERE
  print ('current pointed at', x, y)
  gaussian_mean = 0

  gaussian_std = float(radius) / 2.0 # compute the standard deviation
  #copy_image = image.copy()
  xdim = image.shape[1] # x values
  ydim = image.shape[0] # y values

  xmin = x - radius # compute the neighborhood
  xmax = x + radius
  ymin = y - radius
  ymax = y + radius
  
  #imagec = image.copy()

  for i in range(xmin, xmax):
    for j in range(ymin, ymax):
      distance = np.sqrt(((i-x)**2 + (j-y)**2)) # euclidian distance around the clicked point (a circle)
      if distance <= (radius):
        #exp_part = np.exp(-0.5 * ((distance / gaussian_std) ** 2))
        gaussian_factor = np.exp(-0.5 * ((distance / gaussian_std) ** 2)) # no need to normalized the gaussian
        #gaussian_factor = (1/(gaussian_std * np.sqrt(2 * np.pi))) * exp_part
        mode = editmode(gaussian_factor) # get the factor for additive mode and subtractive mode

        if isFT: # if the graph is in FT
          ftx = wrap(i, xdim) # check for outbound values
          fty = wrap(j, ydim)
          ak = 2*np.real(image[fty, ftx]) # derive for angle value based on "real fourier transform" notes
          bk = 2*np.imag(image[fty, ftx]) # this note is provided by Prof. Stewart
          A = np.sqrt((ak ** 2 + bk ** 2))
          theta = np.arctan2(bk, ak)
          factor = np.log(A + 1)
          newA = np.exp(factor * mode) - 1
          pixel_value = newA * np.cos(theta) + 1j * newA * np.sin(theta) # compute new pixel values

          image[fty, ftx] = pixel_value # update current pixel value and the corresponding symmetric point in the graph
          image[ydim - 1 - fty, xdim - 1 - ftx] = pixel_value

        else:
          if (i >= 0 and i < xdim and j >= 0 and j < ydim): # check for outbound values
            image[j, i] *= mode

#   #pass
def editmode(gaussian): # helper function to detect which mode we want
  if editMode == "s":
    return 1 - gaussian
  return 1 + 0.1*gaussian

# For an image coordinate, if it's < 0 or >= max, wrap the coorindate
# around so that it's in the range [0,max-1].  This is useful in the
# modulatePixels() function when dealing with FT images.

def wrap( val, max ):

  if val < 0:
    return val+max
  elif val >= max:
    return val-max
  else:
    return val