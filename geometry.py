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

def compute_rotation(ip1, ip2, K1, K2):
  """
  Find rotation matrix R such that |r2 - R*r1|^2 is minimised

  Inputs: ip1,ip2=corresponding interest points (2,num_points),
          K1,K2=camera calibration matrices (3,3)

  Outputs: R=rotation matrix R (3,3)

  r1,r2 are corresponding rays (unit normalised camera coordinates) in image 1,2
  """

  # R=H=np.eye(3,3)

  """
  **********************************************************************
  *** TODO: write code to compute 3D rotation between corresponding rays
  **********************************************************************
  """

  # # Get rays first from interest points
  # ray1 = _2d_to_3d(K1, ip1)
  # ray2 = _2d_to_3d(K2, ip2)
  # print('Rays {} {}'.format(ray1[:,4], ray2[:,4]))
  #
  # # Find value of the R
  # # num_points = ray1.shape[1]
  # # c = np.zeros((3,3))
  # # for i in range(num_points):
  # #   ray1_curr = np.expand_dims(ray1[:, i], axis=1)
  # #   ray2_curr = np.transpose(np.expand_dims(ray2[:, i], axis=1))
  # #
  # #   c = c + np.dot(ray1_curr, ray2_curr)
  # #
  # # print('C=')
  # # print(c)
  # print('C=')
  # c = np.dot(ray1, np.transpose(ray2))
  # print(c)
  #
  # u, s, v = np.linalg.svd(c, full_matrices=True)
  # print('SVD {} {} {}'.format(u.shape, s.shape, v.shape))
  # R = np.dot(u, v)
  # # R = u * v
  # print('R {}'.format(R.shape))
  #
  # # H = K2.R.K1inv
  # H = np.dot(K2, np.dot(R, np.linalg.inv(K1)))
  # print('H {}'.format(H))
  #
  # # H = np.dot(np.dot(K2, R),np.linalg.inv(K1))
  # # print('H {}'.format(H))
  r1 = np.dot(np.linalg.inv(K1), hom(ip1))
  r2 = np.dot(np.linalg.inv(K2), hom(ip2))

  c = np.dot(r2, np.transpose(r1))

  u, s, vt = np.linalg.svd(c)

  R = np.dot(u, vt)

  H = np.dot(np.dot(K2, R), np.linalg.inv(K1))

  """
  **********************************************************************
  """
  return R, H

def get_calibration(imshape, fov_degrees):
  """
  Return calibration matrix K given image shape and field of view

  See note on calibration matrix in documentation of K(f, H, W)
  """
  H, W, _ = imshape
  f = max(H,W)/(2*np.tan((fov_degrees/2)*np.pi/180))
  K1 = K(f,H,W)
  return K1

def K(f,H,W):
  """
  Return camera calibration matrix given focal length and image size

  Inputs: f=focal length, H=image height, W=image width all in pixels

  Outputs: K=calibration matrix (3, 3)

  The calibration matrix maps camera coordinates [X,Y,Z] to homogeneous image
  coordinates ~[row,col,1]. X is assumed to point along the positive col direction,
  i.e., incrementing X increments the col dimension in the image

  This is equation on Slide 26
  Camera Coordinates (3D), Real World Coordinates (3D), Image coordinates (2D)

  """
  K1=np.zeros((3,3))
  K1[0,1]=K1[1,0]=f
  K1[0,2]=H/2
  K1[1,2]=W/2
  K1[2,2]=1
  return K1

def hom(p):
  """
  Convert points to homogeneous coordiantes
  """
  ph=np.concatenate((p,np.ones((1,p.shape[1]))))
  return ph

def unhom(ph):
  """
  Convert points from homogeneous to regular coordinates
  """
  p=ph/ph[2,:]
  p=p[0:2,:]
  return p

def _2d_to_3d(K, points):
  """
  hom(points) = K.result
  result = Kinv.home(points)
  :type K: calibration matrix 3x3
  :param points: 2*N points
  :return 3xN matrix
  """
  assert points.shape[0] == 2
  return np.dot(np.linalg.inv(K),hom(points))
