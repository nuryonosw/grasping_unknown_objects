#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import tensorflow as tfw
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse
import rospy
import copy
import sys
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import tf
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PoseStamped, TransformStamped, Quaternion
from moveit_commander import RobotCommander, PlanningSceneInterface
from moveit_msgs.msg import (Constraints, MoveItErrorCodes,
                             OrientationConstraint, PickupAction, PickupGoal,
                             PlaceAction, PlaceGoal, PlanningOptions, PickupResult)
from moveit_msgs.msg import Grasp

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
import scipy
from shapely.geometry import Polygon
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
import random
from gazebo_msgs.srv import GetModelStateRequest, GetModelState
from iki_manipulation.recognized_object import RecognizedObject

pi     = scipy.pi
dot    = scipy.dot
sin    = scipy.sin
cos    = scipy.cos
ar     = scipy.array
bridge = CvBridge()

CLASSES = ('__background__',
           'angle_01', 'angle_02', 'angle_03', 'angle_04', 'angle_05',
           'angle_06', 'angle_07', 'angle_08', 'angle_09', 'angle_10',
           'angle_11', 'angle_12', 'angle_13', 'angle_14', 'angle_15',
           'angle_16', 'angle_17', 'angle_18', 'angle_19')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',),'res101': ('res101_faster_rcnn_iter_110000.ckpt',),'res50': ('res50_faster_rcnn_iter_240000.ckpt',)}
DATASETS= {'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',),'grasp': ('train',)}


class GraspRGB:
    def __init__(self):
        # Define variables
        self.pcd_message = PointCloud2()
        self.rgb_message = Image()
        self.camera_info = CameraInfo()

        cfg.TEST.HAS_RPN = True  # Use RPN for proposals
        self.args = self.parse_args()

        # model path
        self.demonet = self.args.demo_net
        self.dataset = self.args.dataset
        self.tfmodel = os.path.join('output', self.demonet, DATASETS[self.dataset][0], 'default',
                                  NETS[self.demonet][0])


        if not os.path.isfile(self.tfmodel + '.meta'):
            raise IOError(('{:s} not found.\nDid you download the proper networks from '
                           'our server and place them properly?').format(self.tfmodel + '.meta'))

        # set config
        self.tfconfig = tfw.ConfigProto(allow_soft_placement=True)
        self.tfconfig.gpu_options.allow_growth=True

        # init session
        self.sess = tfw.Session(config=self.tfconfig)

        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(1200.0))


    def Rotate2D(self, pts, cnt, ang=scipy.pi/4):
        '''pts = {} Rotates points(nx2) about center cnt(2) by angle ang(1) in radian'''
        return dot(pts-cnt,ar([[cos(ang),sin(ang)],[-sin(ang),cos(ang)]]))+cnt

    def vis_detections(self, ax, im, class_name, dets, thresh=0.5):
        self.camera_info = rospy.wait_for_message('/xtion/depth_registered/camera_info', CameraInfo)
        fx = self.camera_info.K[0]
        cx = self.camera_info.K[2]
        fy = self.camera_info.K[4]
        cy = self.camera_info.K[5]

        """Draw detected bounding boxes."""
        inds = np.where(dets[:, -1] >= thresh)[0]
        if len(inds) == 0:
            return

        im = im[:, :, (2, 1, 0)]
        #fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(im, aspect='equal')
        x = []
        y = []
        z = []
        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]

            # plot rotated rectangles
            pts = ar([[bbox[0],bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]])
            cnt = ar([(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2])
            angle = int(class_name[6:])
            r_bbox = self.Rotate2D(pts, cnt, -pi/2-pi/20*(angle-1))
            pred_label_polygon = Polygon([(r_bbox[0,0],r_bbox[0,1]), (r_bbox[1,0], r_bbox[1,1]), (r_bbox[2,0], r_bbox[2,1]), (r_bbox[3,0], r_bbox[3,1])])
            pred_x, pred_y = pred_label_polygon.exterior.xy

            pred_xt = np.round(pred_x).astype(np.int)
            pred_yt = np.round(pred_y).astype(np.int)

            plt.plot(pred_x[0:2],pred_y[0:2], color='k', alpha = 0.7, linewidth=1, solid_capstyle='round', zorder=2)
            plt.plot(pred_x[1:3],pred_y[1:3], color='r', alpha = 0.7, linewidth=3, solid_capstyle='round', zorder=2)
            plt.plot(pred_x[2:4],pred_y[2:4], color='k', alpha = 0.7, linewidth=1, solid_capstyle='round', zorder=2)
            plt.plot(pred_x[3:5],pred_y[3:5], color='r', alpha = 0.7, linewidth=3, solid_capstyle='round', zorder=2)




    def detect(self, sess, net, im):
        """Detect object classes in an image using pre-computed object proposals."""
        # Detect all object classes and regress object bounds
        self.timer = Timer()
        self.timer.tic()
        self.scores, self.boxes = im_detect(sess, net, im)

        self.timer.toc()
        print('Detection took {:.3f}s for {:d} object proposals'.format(self.timer.total_time, self.boxes.shape[0]))

        self.fig, self.ax = plt.subplots(figsize=(12, 12))
        # Visualize detections for each class
        CONF_THRESH = 0.1
        NMS_THRESH = 0.3
        for cls_ind, cls in enumerate(CLASSES[1:]):
            cls_ind += 1 # because we skipped background
            cls_boxes = self.boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = self.scores[:, cls_ind]
            self.dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
            self.keep = nms(self.dets, NMS_THRESH)
            self.dets = self.dets[self.keep, :]
            self.vis_detections(self.ax, im, cls, self.dets, thresh=CONF_THRESH)

        plt.axis('off')
        plt.tight_layout()
        #save result
        savepath = './data/demo/results_all_cls/' + str(random.randint(1,1001)) + '.png'
        plt.savefig(savepath)
        plt.draw()


    def grasp(self):
        # load network
        if self.demonet == 'vgg16':
            self.net = vgg16(batch_size=1)
        elif self.demonet == 'res101':
            self.net = resnetv1(batch_size=1, num_layers=101)
        elif self.demonet == 'res50':
            self.net = resnetv1(batch_size=1, num_layers=50)
        else:
            raise NotImplementedError
        self.net.create_architecture(self.sess, "TEST", 20,
                              tag='default', anchor_scales=[8, 16, 32])
        self.saver = tfw.train.Saver()
        self.saver.restore(self.sess, self.tfmodel)

        print('Loaded network {:s}'.format(self.tfmodel))
        # Get Scene
        self.rgb_message = rospy.wait_for_message('/xtion/rgb/image_raw', Image)
        self.pcd_message = rospy.wait_for_message('/xtion/depth_registered/points', PointCloud2)
        self.rgb = bridge.imgmsg_to_cv2(self.rgb_message)
        self.detect(self.sess, self.net, self.rgb)

        plt.show()

    def parse_args(self):
        """Parse input arguments."""
        parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN')
        parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                            choices=NETS.keys(), default='res101')
        parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                            choices=DATASETS.keys(), default='pascal_voc_0712')
        args = parser.parse_args()

        return args

if __name__ == '__main__':
    while not rospy.is_shutdown():
        rospy.init_node('detect')
        call = GraspRGB()
        call.grasp()
