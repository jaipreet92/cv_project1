# Copyright 2017 Google Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import scipy.ndimage.filters as filters
from scipy.ndimage import map_coordinates
from matplotlib.patches import Circle
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from cyvlfeat import sift 


import im_util

class InterestPointExtractor:
  """
  Class to extract interest points from an image
  """
  def __init__(self):
    self.params={}
    self.params['border_pixels'] = 30
    self.params['strength_threshold_percentile']=95
    self.params['supression_radius_frac']=0.01

  def set_border_pixels(self, border_pixels_new):
    self.params['border_pixels'] = border_pixels_new 

  def find_interest_points(self, img):
    """
    Find interest points in greyscale image img

    Inputs: img=greyscale input image (H, W, 1)

    Outputs: ip=interest points of shape (2, N)
    """
    # ip_fun = self.corner_function(img)
    # row, col = self.find_local_maxima(ip_fun)
    # ip = np.stack((row,col))

    frames,desc=sift.sift(img,
      compute_descriptor=True,
      n_levels=1,
      peak_thresh=0.1,
      edge_thresh=10.0
      )
    ip=(frames.T)[0:2,:]
    desc=desc.astype(np.float)
    return ip

  def corner_function(self, img):
    """
    Compute corner strength function in image im

    Inputs: img=grayscale input image (H, W, 1)

    Outputs: ip_fun=interest point strength function (H, W, 1)
    """

    H, W, _ = img.shape

    # FORNOW: random interest point function
    # ip_fun = np.random.randn(H, W, 1)

    """
    **********************************************************
    *** TODO: write code to compute a corner strength function
    **********************************************************
    """
    # #Harris Corners approach
    # print('Using Harris Corner')
    # Ix, Iy = im_util.compute_gradients(img)
    # sigma = 4.0
    # Ix2 = im_util.convolve_gaussian(np.square(Ix), sigma) # Has a value at every pixel, implement the w(x,y) as a convulution (Gaussian, size of the kernel is arbitrary)
    # Iy2 = im_util.convolve_gaussian(np.square(Iy), sigma) #
    # Ixy = im_util.convolve_gaussian(Ix*Iy, sigma)

    # det = (Ix2 * Iy2) - (Ixy**2)
    # trace = Ix2 + Iy2
    # k = 0.6 #0.4 -0.6
    # ip_fun = det - k*(trace**2)

    # DoG
    print('Using DoG')

    ip_fun = im_util.convolve_gaussian(img, 6.5) - im_util.convolve_gaussian(img, 4.0)

    # print('CORNER_FUNCTION: Ix2: {} Iy2: {} Ixy: {}, ip_fun: {}'.format(Ix2.shape, Iy2.shape, Ixy.shape, ip_fun.shape))
    """
    **********************************************************
    """
    
    return ip_fun

  def find_local_maxima(self, ip_fun):
    """
    Find local maxima in interest point strength function

    Inputs: ip_fun=corner strength function (H, W, 1)

    Outputs: row,col=coordinates of interest points
    """

    H, W, _ = ip_fun.shape

    # radius for non-maximal suppression
    suppression_radius_pixels = int(self.params['supression_radius_frac']*max(H, W))

    # minimum of strength function for corners
    strength_threshold=np.percentile(ip_fun, self.params['strength_threshold_percentile'])

    # don't return interest points within border_pixels of edge
    border_pixels = self.params['border_pixels']

    # row and column coordinates of interest points
    row = []
    col = []

    # FORNOW: random row and column coordinates
    # row = np.random.randint(0,H,100)
    # col = np.random.randint(0,W,100)

    """
    ***************************************************
    *** TODO: write code to find local maxima in ip_fun
    ***************************************************

    Hint: try scipy filters.maximum_filter with im_util.disc_mask
    """
    footprint = im_util.disc_mask(suppression_radius_pixels)
    print('Footprint: {}'.format(footprint.shape))
    mx = filters.maximum_filter(ip_fun, footprint.shape, footprint)
    
    print('Threshold: {}'.format(strength_threshold))
    row, col, _ = np.where((mx > strength_threshold) & (mx == ip_fun))

    print('Num interest points: {} Min/Max Row {} {}Min/Max Col {} {}'.format(len(row), row.min(), row.max(), col.min(), col.max()) )
  

    # Remove border pixels
    rows_bp = []
    cols_bp = []
    total = len(row)
    for idx in range(total):
        curr_row = row[idx]
        curr_col = col[idx]
        if curr_row >= border_pixels and curr_col >=border_pixels and (curr_col <= W - border_pixels) and (curr_row <= H - border_pixels):
          rows_bp.append(curr_row)
          cols_bp.append(curr_col)

    row = np.asarray(rows_bp)
    col = np.asarray(cols_bp)

    print('Num interest points after border_pixels: {} {}'.format(len(row), len(col)) )

    """
    ***************************************************
    """

    return row, col

class DescriptorExtractor:
  """
  Extract descriptors around interest points
  """
  def __init__(self):
    self.params={}
    self.params['patch_size']=21
    self.params['ratio_threshold']=.85

  def set_patch_size(self, new_patch_size):
    self.params['patch_size'] = new_patch_size

  def set_ratio_threshold(self, new_ratio_threshold):
    self.params['ratio_threshold'] = new_ratio_threshold

  def get_brief_gaussian_descriptors(self, img, ip):
    """
    Extact descriptors from grayscale image img at interest points ip

    Inputs: img=grayscale input image (H, W, 1)
            ip=interest point coordinates (2, N)

    Returns: descriptors=vectorized descriptors (N, num_dims)
    """
    patch_size=self.params['patch_size']
    patch_size_div2=int(patch_size/2)
    num_dims=patch_size**2

    # The indexes in the patch that we will sample.
    brief_uniform_random_patch_size = 15
    num_dims = brief_uniform_random_patch_size**2
    idxs = np.random.choice(patch_size**2, num_dims, replace=False)

    print('Patch size: {} Num_dims: {} , ip {}, img {}'.format(patch_size, num_dims, ip.shape, img.shape))

    H,W,_=img.shape
    num_ip=ip.shape[1]
    descriptors=np.zeros((num_ip,num_dims))

    for i in range(num_ip):
      row=ip[0,i]
      col=ip[1,i]

      # FORNOW: random image patch
      # patch=np.random.randn(patch_size,patch_size)

      """
      ******************************************************
      *** TODO: write code to extract descriptor at row, col
      ******************************************************
      """
      row_start = 0 if row-patch_size_div2 < 0 else row-patch_size_div2
      row_end = H if row+patch_size_div2+1 > H else row+patch_size_div2+1
      col_start = 0 if col-patch_size_div2 < 0 else col-patch_size_div2
      col_end = W if col+patch_size_div2+1 > W else col+patch_size_div2+1
      patch = img[row_start:row_end, col_start:col_end ]

      """
      ******************************************************
      """
      patch_flat = np.reshape(patch, patch_size**2)

      descriptors[i, :]=patch_flat[idxs]

    # normalise descriptors to 0 mean, unit length
    mn=np.mean(descriptors,1,keepdims=True)
    sd=np.std(descriptors,1,keepdims=True)
    small_val = 1e-6
    descriptors = (descriptors-mn)/(sd+small_val)

    return descriptors

  def get_pca_sift_descriptors(self, img, ip):
    """
    Extact descriptors from grayscale image img at interest points ip

    Inputs: img=grayscale input image (H, W, 1)
            ip=interest point coordinates (2, N)

    Returns: descriptors=vectorized descriptors (N, num_dims)
    """
    # Fixed for PCA-SIFT
    patch_size=39
    patch_size_div2=int(patch_size/2)

    # Fixed for PCA-SIFT
    num_dims=3042

    print('Patch size: {} Num_dims: {} , ip {}, img {}'.format(patch_size, num_dims, ip.shape, img.shape))

    H,W,_=img.shape
    num_ip=ip.shape[1]
    descriptors=np.zeros((num_ip,num_dims))

    for i in range(num_ip):
      row=ip[0,i]
      col=ip[1,i]

      # FORNOW: random image patch
      # patch=np.random.randn(patch_size,patch_size)

      """
      ******************************************************
      *** TODO: write code to extract descriptor at row, col
      ******************************************************
      """
      row_start = 0 if row-patch_size_div2 < 0 else row-patch_size_div2
      row_end = H if row+patch_size_div2+1 > H else row+patch_size_div2+1
      col_start = 0 if col-patch_size_div2 < 0 else col-patch_size_div2
      col_end = W if col+patch_size_div2+1 > W else col+patch_size_div2+1
      patch = img[row_start:row_end, col_start:col_end ]

      """
      ******************************************************
      """
      ix, iy = im_util.compute_gradients(patch)
      ix = np.squeeze(ix)
      iy = np.squeeze(iy)
      patch = np.reshape(patch, patch_size**2)
      pca_sift_patch = np.ndarray.flatten(np.array([np.ndarray.flatten(ix),np.ndarray.flatten(ix)]))

      # TODO: Do PCA
      descriptors[i, :]=pca_sift_patch

    
    # Implement PCA on descripts
    pca = PCA(n_components=36)
    descriptors = pca.fit_transform(descriptors)
    print('Patch Gradient: {} {}'.format(pca_sift_patch.shape, descriptors.shape))

    ## END

    # normalise descriptors to 0 mean, unit length
    mn=np.mean(descriptors,1,keepdims=True)
    sd=np.std(descriptors,1,keepdims=True)
    small_val = 1e-6
    descriptors = (descriptors-mn)/(sd+small_val)

    return descriptors

  def get_brief_uniform_random_descriptors(self, img, ip):
    """
    Extact descriptors from grayscale image img at interest points ip

    Inputs: img=grayscale input image (H, W, 1)
            ip=interest point coordinates (2, N)

    Returns: descriptors=vectorized descriptors (N, num_dims)
    """
    patch_size=self.params['patch_size']
    patch_size_div2=int(patch_size/2)
    num_dims=patch_size**2

    # The indexes in the patch that we will sample.
    brief_uniform_random_patch_size = 19
    num_dims = brief_uniform_random_patch_size**2
    # We will use the indexes for all patches, to ensure uniformity.
    idxs = np.random.choice(patch_size**2, num_dims, replace=False)

    print('Patch size: {} Num_dims: {} , ip {}, img {}'.format(patch_size, num_dims, ip.shape, img.shape))

    H,W,_=img.shape
    num_ip=ip.shape[1]
    descriptors=np.zeros((num_ip,num_dims))

    for i in range(num_ip):
      row=ip[0,i]
      col=ip[1,i]

      # FORNOW: random image patch
      # patch=np.random.randn(patch_size,patch_size)

      """
      ******************************************************
      *** TODO: write code to extract descriptor at row, col
      ******************************************************
      """
      row_start = 0 if row-patch_size_div2 < 0 else row-patch_size_div2
      row_end = H if row+patch_size_div2+1 > H else row+patch_size_div2+1
      col_start = 0 if col-patch_size_div2 < 0 else col-patch_size_div2
      col_end = W if col+patch_size_div2+1 > W else col+patch_size_div2+1
      patch = img[row_start:row_end, col_start:col_end ]

      """
      ******************************************************
      """
      patch_flat = np.reshape(patch, patch_size**2)

      descriptors[i, :]=patch_flat[idxs]

    # normalise descriptors to 0 mean, unit length
    mn=np.mean(descriptors,1,keepdims=True)
    sd=np.std(descriptors,1,keepdims=True)
    small_val = 1e-6
    descriptors = (descriptors-mn)/(sd+small_val)

    return descriptors

  def get_descriptors(self, img, ip):
    """
    Extact descriptors from grayscale image img at interest points ip

    Inputs: img=grayscale input image (H, W, 1)
            ip=interest point coordinates (2, N)

    Returns: descriptors=vectorized descriptors (N, num_dims)
    """
    # patch_size=self.params['patch_size']
    # patch_size_div2=int(patch_size/2)
    # num_dims=patch_size**2
    #
    # print('Patch size: {} Num_dims: {} , ip {}, img {}'.format(patch_size, num_dims, ip.shape, img.shape))
    #
    # H,W,_=img.shape
    # num_ip=ip.shape[1]
    #
    # sample_spacing = 2
    # if sample_spacing > 1:
    #   # We want to keep the size of the image as a square, since the plotting method squares them.
    #   num_dims = int(np.sqrt(patch_size**2 / sample_spacing))**2
    # else:
    #   num_dims = patch_size**2
    # descriptors=np.zeros((num_ip,num_dims))
    #
    #
    # for i in range(num_ip):
    #   row=ip[0,i]
    #   col=ip[1,i]
    #
    #   # FORNOW: random image patch
    #   # patch=np.random.randn(patch_size,patch_size)
    #
    #   """
    #   ******************************************************
    #   *** TODO: write code to extract descriptor at row, col
    #   ******************************************************
    #   """
    #   row_start = 0 if row-patch_size_div2 < 0 else row-patch_size_div2
    #   row_end = H if row+patch_size_div2+1 > H else row+patch_size_div2+1
    #   col_start = 0 if col-patch_size_div2 < 0 else col-patch_size_div2
    #   col_end = W if col+patch_size_div2+1 > W else col+patch_size_div2+1
    #   patch = img[row_start:row_end, col_start:col_end ]
    #
    #   """
    #   ******************************************************
    #   """
    #   if sample_spacing > 1:
    #     patch = np.ndarray.flatten(patch)
    #     patch = patch[::sample_spacing]
    #     patch = patch[:num_dims]
    #     descriptors[i, :]= patch
    #   else:
    #     descriptors[i, :]=np.reshape(patch,num_dims)
    #
    # # normalise descriptors to 0 mean, unit length
    # mn=np.mean(descriptors,1,keepdims=True)
    # sd=np.std(descriptors,1,keepdims=True)
    # small_val = 1e-6
    # descriptors = (descriptors-mn)/(sd+small_val)
    #
    # return descriptors

    frames,desc=sift.sift(img,
      compute_descriptor=True,
      n_levels=1,
      peak_thresh=0.1,
      edge_thresh=10.0
      )
    ip=(frames.T)[0:2,:]
    desc=desc.astype(np.float)
    return desc

  def compute_distances(self, desc1, desc2):
    """
    Compute distances between descriptors

    Inputs: desc1=descriptor array (N1, num_dims)
            desc2=descriptor array (N2, num_dims)

    Returns: dists=array of distances (N1,N2)
    """
    N1,num_dims=desc1.shape
    N2,num_dims=desc2.shape

    ATB=np.dot(desc1,desc2.T)
    AA=np.sum(desc1*desc1,1)
    BB=np.sum(desc2*desc2,1)

    dists=-2*ATB+np.expand_dims(AA,1)+BB

    return dists

  def match_descriptors(self, desc1, desc2):
    """
    Find nearest neighbour matches between descriptors

    Inputs: desc1=descriptor array (N1, num_dims)
            desc2=descriptor array (N2, num_dims)

    Returns: match_idx=nearest neighbour index for each desc1 (N1)
    """
    dists=self.compute_distances(desc1, desc2)

    match_idx=np.argmin(dists,1)

    return match_idx

  def match_ratio_test(self, desc1, desc2):
    """
    Find nearest neighbour matches between descriptors
    and perform ratio test

    Inputs: desc1=descriptor array (N1, num_dims)
            desc2=descriptor array (N2, num_dims)

    Returns: match_idx=nearest neighbour inded for each desc1 (N1)
             ratio_pass=whether each match passes ratio test (N1)
    """
    N1,num_dims=desc1.shape

    dists=self.compute_distances(desc1, desc2)

    sort_idx=np.argsort(dists,1)

    #match_idx=np.argmin(dists,1)
    match_idx=sort_idx[:,0]

    d1NN=dists[np.arange(0,N1),sort_idx[:,0]]
    d2NN=dists[np.arange(0,N1),sort_idx[:,1]]

    ratio_threshold=self.params['ratio_threshold']
    ratio_pass=(d1NN<ratio_threshold*d2NN)

    return match_idx,ratio_pass

def draw_interest_points_ax(ip, ax):
  """
  Draw interest points ip on axis ax
  """
  for row,col in zip(ip[0,:],ip[1,:]):
    circ1 = Circle((col,row), 5)
    circ1.set_color('black')
    circ2 = Circle((col,row), 3)
    circ2.set_color('white')
    ax.add_patch(circ1)
    ax.add_patch(circ2)

def draw_interest_points_file(im, ip, filename):
  """
  Draw interest points ip on image im and save to filename
  """
  fig,ax = im_util.image_figure(im)
  draw_interest_points_ax(ip, ax)
  fig.savefig(filename)
  plt.close(fig)

def draw_matches_ax(ip1, ipm, ax1, ax2):
  """
  Draw matches ip1, ipm on axes ax1, ax2
  """
  for r1,c1,r2,c2 in zip(ip1[0,:], ip1[1,:], ipm[0,:], ipm[1,:]):
    rand_colour=np.random.rand(3,)

    circ1 = Circle((c1,r1), 5)
    circ1.set_color('black')
    circ2 = Circle((c1,r1), 3)
    circ2.set_color(rand_colour)
    ax1.add_patch(circ1)
    ax1.add_patch(circ2)

    circ3 = Circle((c2,r2), 5)
    circ3.set_color('black')
    circ4 = Circle((c2,r2), 3)
    circ4.set_color(rand_colour)
    ax2.add_patch(circ3)
    ax2.add_patch(circ4)

def draw_matches_file(im1, im2, ip1, ipm, filename):
  """
  Draw matches ip1, ipm on images im1, im2 and save to filename
  """
  H1,W1,B1=im1.shape
  H2,W2,B2=im2.shape

  im3 = np.zeros((max(H1,H2),W1+W2,3))
  im3[0:H1,0:W1,:]=im1
  im3[0:H2,W1:(W1+W2),:]=im2

  fig,ax = im_util.image_figure(im3)
  col_offset=W1

  for r1,c1,r2,c2 in zip(ip1[0,:], ip1[1,:], ipm[0,:], ipm[1,:]):
    rand_colour=np.random.rand(3,)

    circ1 = Circle((c1,r1), 5)
    circ1.set_color('black')
    circ2 = Circle((c1,r1), 3)
    circ2.set_color(rand_colour)
    ax.add_patch(circ1)
    ax.add_patch(circ2)

    circ3 = Circle((c2+col_offset,r2), 5)
    circ3.set_color('black')
    circ4 = Circle((c2+col_offset,r2), 3)
    circ4.set_color(rand_colour)
    ax.add_patch(circ3)
    ax.add_patch(circ4)

  fig.savefig(filename)
  plt.close(fig)

def plot_descriptors(desc,plt):
  """
  Plot a random set of descriptor patches
  """
  num_ip,num_dims = desc.shape
  patch_size = int(np.sqrt(num_dims))

  N1,N2=2,8
  figsize0=plt.rcParams['figure.figsize']
  plt.rcParams['figure.figsize'] = (16.0, 4.0)
  for i in range(N1):
    for j in range(N2):
      ax=plt.subplot(N1,N2,i*N2+j+1)
      rnd=np.random.randint(0,num_ip)
      desc_im=np.reshape(desc[rnd,:],(patch_size,patch_size))
      plt.imshow(im_util.grey_to_rgb(im_util.normalise_01(desc_im)))
      plt.axis('off')

  plt.rcParams['figure.figsize']=figsize0

def plot_matching_descriptors(desc1,desc2,desc1_id,desc2_id,plt):
  """
  Plot a random set of matching descriptor patches
  """
  num_inliers=desc1_id.size
  num_ip,num_dims = desc1.shape
  patch_size=int(np.sqrt(num_dims))

  figsize0=plt.rcParams['figure.figsize']

  N1,N2=1,8
  plt.rcParams['figure.figsize'] = (16.0, N1*4.0)

  for i in range(N1):
    for j in range(N2):
      rnd=np.random.randint(0,num_inliers)

      desc1_rnd=desc1_id[rnd]
      desc2_rnd=desc2_id[rnd]

      desc1_im=np.reshape(desc1[desc1_rnd,:],(patch_size,patch_size))
      desc2_im=np.reshape(desc2[desc2_rnd,:],(patch_size,patch_size))

      ax=plt.subplot(2*N1,N2,2*i*N2+j+1)
      plt.imshow(im_util.grey_to_rgb(im_util.normalise_01(desc1_im)))
      plt.axis('off')
      ax=plt.subplot(2*N1,N2,2*i*N2+N2+j+1)
      plt.imshow(im_util.grey_to_rgb(im_util.normalise_01(desc2_im)))
      plt.axis('off')

  plt.rcParams['figure.figsize'] = figsize0
