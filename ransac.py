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

import os.path
import numpy as np
from time import time

import im_util
import interest_point
import geometry


class RANSAC:
    """
  Find 2-view consistent matches using RANSAC
  """

    def __init__(self):
        self.params = {}
        self.params['num_iterations'] = 25000
        self.params['inlier_dist'] = 10
        self.params['min_sample_dist'] = 2

    def consistent(self, H, p1, p2):
        """
        Find interest points that are consistent with 2D transform H

        Inputs: H=homography matrix (3,3)
                p1,p2=corresponding points in images 1,2 of shape (2, N)

        Outputs: cons=list of inliers indicated by true/false (num_points)

        Assumes that H maps from 1 to 2, i.e., hom(p2) ~= H hom(p1)
        """

        cons = np.full((p1.shape[1]), False, dtype=bool)
        inlier_dist = self.params['inlier_dist']

        """
        ************************************************
        *** TODO: write code to check consistency with H
        ************************************************
        """
        # dist = np.linalg.norm(p2 - p1, axis=0)

        for i in range(0, p1.shape[1]):
            X = np.array([p1[0, i], p1[1, i], 1], ndmin=2).transpose()
            XX = np.dot(H, X)
            d = np.sqrt(((p2[0, i] - XX[0]) ** 2) + ((p2[1, i] - XX[1]) ** 2))
            cons[i] = (d < inlier_dist)

        """
        ************************************************
        """

        return cons

    def compute_similarity(self, p1, p2):
        """
        Compute similarity transform between pairs of points

        Input: p1,p2=arrays of coordinates (2, 2)

        Output: Similarity matrix S (3, 3)

        Assume S maps from 1 to 2, i.e., hom(p2) = S hom(p1)
        """

        S = np.eye(3, 3)

        """
        ****************************************************
        *** TODO: write code to compute similarity transform
        ****************************************************
        """

        # a*x1 - b*y1 + tx = x1'
        # b*x1 + a*y1 + ty = y1'
        # a*x2 - b*y2 + tx = x2'
        # b*x2 + a*y2 + ty = y2'
        x1 = p1[0,0]
        y1 = p1[1,0]
        x2 = p1[0,1]
        y2 = p1[1,1]

        x1_ = p2[0,0]
        y1_ = p2[1,0]
        x2_ = p2[0,1]
        y2_ = p2[1,1]

        lhs = np.array([
            [x1, -y1, 1, 0],
            [x1, y1, 0 , 1],
            [x2, -y2, 1, 0],
            [x2, y2, 0, 1],
        ])

        if np.linalg.det(lhs) == 0.0:
            return S

        rhs = np.array([x1_, y1_, x2_, y2_])

        sol = np.linalg.solve(lhs, rhs)
        # print('Solution: {}'.format(sol.shape))

        # Check that the solution is correct
        # print(np.allclose(np.dot(lhs, sol), rhs))

        S = np.array([
            [sol[0], -sol[1], sol[2]],
            [sol[1], sol[0], sol[3]],
            [0,0,1]
        ])
        """
        ****************************************************
        """

        return S

    def ransac_similarity(self, ip1, ipm):
        """
        Find 2-view consistent matches under a Similarity transform

        Inputs: ip1=interest points (2, num_points)
                ipm=matching interest points (2, num_points)
                ip[0,:]=row coordinates, ip[1, :]=column coordinates

        Outputs: S_best=Similarity matrix (3,3)
                 inliers_best=list of inliers indicated by true/false (num_points)
        """
        S_best = np.eye(3, 3)
        inliers_best = []

        """
        *****************************************************
        *** TODO: use ransac to find a similarity transform S
        *****************************************************
        """
        num_iterations = self.params['num_iterations']
        min_sample = self.params['min_sample_dist']
        assert ip1.shape[1] == ipm.shape[1]
        num_points = ip1.shape[1]

        for i in range(num_iterations):
            sample_idxs = np.random.randint(low=0, high=num_points, size=min_sample)
            # print('Sample idxs: {} at iteration: {}'.format(sample_idxs, i))
            s_current = self.compute_similarity(ip1[:, sample_idxs], ipm[:, sample_idxs])
            inliers_current = self.consistent(s_current, ip1, ipm)
            num_consistent_current = np.sum(inliers_current)

            if num_consistent_current > np.sum(inliers_best):
                print('Updating S_Best at iteration {} ; num_inliers {}/{}; idx {}'.format(i, num_consistent_current,
                                                                                           num_points, sample_idxs))
                inliers_best = inliers_current
                S_best = s_current

        """
        *****************************************************
        """

        return S_best, inliers_best
