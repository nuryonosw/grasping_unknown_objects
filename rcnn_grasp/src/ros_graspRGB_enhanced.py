#!/usr/bin/env python
# Import necessary libraries and ROS messages.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

import math
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
import moveit_commander
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
import pcl
import pcl.pcl_visualization
from tf.transformations import quaternion_about_axis, rotation_matrix, quaternion_from_matrix, quaternion_multiply
import actionlib
from play_motion_msgs.msg import PlayMotionAction, PlayMotionGoal

# Initialize global constants.
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
        # Initialize variables
        self.pcd_message = PointCloud2()
        self.rgb_message = Image()
        self.camera_info = CameraInfo()
        self.ox = []
        self.oy = []
        self.oz = []
        self.ow = []
        self.trans_x = []
        self.trans_y = []
        self.trans_z = []
        self.trans_pre_x = []
        self.trans_pre_y = []
        self.trans_pre_z = []
        # Init MoveIt
        self.robot = RobotCommander()
        self.scene = PlanningSceneInterface()
        self.arm = self.robot.get_group('arm_torso')
        self.group = moveit_commander.MoveGroupCommander("arm_torso")
        self.gripper = self.robot.get_group('gripper')
        self.arm.set_planning_time(0.0)
        self.arm.set_planner_id("RRTConnectkConfigDefault")
        self.arm.set_pose_reference_frame("base_footprint")
        self.arm.allow_replanning(False)

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

        self.scene_object = RecognizedObject("part")

    def get_object(self):
        # Use gazebo/get_model_state service to get object position.
        rospy.wait_for_service('/gazebo/get_model_state')
        self.target_service = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.target_request = GetModelStateRequest()
        # Specify name of the object to get position.
        self.target_request.model_name = 'pringles'
        self.target_request.relative_entity_name = 'base_footprint'
        self.target_result = self.target_service(self.target_request)
        tmpPoseStamped = PoseStamped()
        tmpPoseStamped.header.frame_id = self.target_result.header.frame_id
        tmpPoseStamped.header.stamp = rospy.Time(0)
        tmpPoseStamped.pose = self.target_result.pose
        self.scene_object.pose = tmpPoseStamped
        # Give object dimensions to add to planning scene.
        self.scene_object.dimensions = (0.04, 0.04, 0.25)
        self.add_pickup_object(self.scene_object)

    def move_to(self, pose_stamped):
        # Function to move arm to a pose in the cartesian space.
        self.arm.set_start_state_to_current_state()
        self.arm.set_pose_target(pose_stamped)
        result = self.arm.go(wait=True)
        return result

    def move_to_linear(self, target_pose_stamped, jump_threshold = 0.0, velocity_scaling_factor = 1.0, min_distance = 0.0):
        # Function to move arm to a pose in cartesian space in a straight line.
        eef_step = 0.01

        rospy.loginfo("Jump threshold: " + str(jump_threshold))
        rospy.loginfo("Endeffector step size in m: " + str(eef_step))

        if target_pose_stamped is None:
            rospy.logerr("pose stamped given is None.")
            return MoveItErrorCodes.FAILURE

        self.arm.set_start_state_to_current_state()

        waypoints = []

        target_pose_stamped_base_link = target_pose_stamped

        rospy.logwarn("Move linear to position {}".format(target_pose_stamped_base_link))

        waypoints.append(target_pose_stamped_base_link.pose)

        (plan, fraction) = self.arm.compute_cartesian_path(waypoints, eef_step, jump_threshold)

        plan = self.arm.retime_trajectory(self.robot.get_current_state(), plan, velocity_scaling_factor)

        rospy.loginfo("Cartesian path fraction: " + str(fraction))
        rospy.loginfo("Cartesian path minimum distance: " + str(min_distance))

        if fraction < min_distance:
            rospy.logerr("Cartesian path fraction was below 1.0 it was: " + str(fraction))
            return False
        result = self.arm.execute(plan,wait=True)
        rospy.loginfo("Move result: " + str(result))
        return result

    def _transform_pose_to_frame(self, pose_stamped, target_frame):
        # Function to transform a pose from one frame to another.
        pose_stamped.header.frame_id = pose_stamped.header.frame_id.strip('/')
        target_frame.strip('/')

        rospy.logdebug("Transform pose \n{}\n from frame {} to frame {}".format(pose_stamped, pose_stamped.header.frame_id, target_frame))

        transform = self.tf_buffer.lookup_transform(target_frame, pose_stamped.header.frame_id, rospy.Time(0), rospy.Duration(1.0))

        pose_stamped = tf2_geometry_msgs.do_transform_pose(pose_stamped, transform)

        rospy.logdebug("Transformed pose: \n{}".format(pose_stamped))

        return pose_stamped

    def add_pickup_object(self, recognized_object):
        # Function to add an object to the planning scene.
        pose_stamped = copy.deepcopy(recognized_object.pose)

        self.scene.add_box(recognized_object.name, pose_stamped,
                           (recognized_object.dimensions[0], recognized_object.dimensions[1],
                            recognized_object.dimensions[2]))

        rospy.loginfo("Object {} added to planning scene".format(recognized_object.name))

        rospy.logdebug("Object {} added with pose \n{} \nand dimensions {}".format(recognized_object.name, pose_stamped,
                                                                                   recognized_object.dimensions))

    def transform_pose(self, p):
        # Function to transform grasp from camera frame to base frame.
        listener = tf.TransformListener()
        try:
            listener.waitForTransform('/base_footprint', '/xtion_rgb_optical_frame', rospy.Time(0), rospy.Duration(5))
            (trans, rot) = listener.lookupTransform('/base_footprint', '/xtion_rgb_optical_frame', rospy.Time(0))
        except Exception as e:
            rospy.logerr('#'*20+"\n\n"+str(e))
        t = TransformStamped()
        t.transform.translation.x = trans[0]
        t.transform.translation.y = trans[1]
        t.transform.translation.z = trans[2]
        t.transform.rotation.x = rot[0]
        t.transform.rotation.y = rot[1]
        t.transform.rotation.z = rot[2]
        t.transform.rotation.w = rot[3]
        q=tf2_geometry_msgs.do_transform_pose(p,t)
        q.header.frame_id = 'base_footprint'
        return q



    def Rotate2D(self, pts, cnt, ang=scipy.pi/4):
        '''pts = {} Rotates points(nx2) about center cnt(2) by angle ang(1) in radian'''
        return dot(pts-cnt,ar([[cos(ang),sin(ang)],[-sin(ang),cos(ang)]]))+cnt

    def vis_detections(self, ax, im, class_name, dets, thresh=0.5):
        """Draw detected bounding boxes."""
        inds = np.where(dets[:, -1] >= thresh)[0]
        if len(inds) == 0:
            return

        im = im[:, :, (2, 1, 0)]
        ax.imshow(im, aspect='equal')
        x = []
        y = []
        z = []
        gen = pc2.read_points(self.pcd_message, skip_nans=False, field_names=("x", "y", "z"))
        gen_list = list(gen)
        pc = pc2.read_points(self.pcd_message, skip_nans=True, field_names=("x", "y", "z"))
        pc_list = []
        for p in pc:
            pc_list.append( [p[0],p[1],p[2]] )

        for i in range(20):
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
            # Get grasp pose and angle from detections and compute orientations based on surface normals.
            centre_x = (pred_xt[0] + pred_xt[1]) / 2
            centre_y = (pred_yt[1] + pred_yt[2]) / 2
            centre_x = np.round(centre_x).astype(np.int)
            centre_y = np.round(centre_y).astype(np.int)
            angle = (angle * np.pi) / 180 + np.pi/2
            arrayPos = centre_y*self.pcd_message.width + centre_x
            arrayPosX = arrayPos
            arrayPosY = arrayPos + 4
            arrayPosZ = arrayPos + 8
            if np.isnan(gen_list[arrayPosX][0]):
                break
            for n in range(len(pc_list)):
                if pc_list[n][0] == gen_list[arrayPosX][0]:
                    pc_arrayPos = n
                    break

            self.cloud = pcl.PointCloud()
            self.cloud.from_list(pc_list)
            ne = self.cloud.make_NormalEstimation()
            ne.set_RadiusSearch(0)
            ne.set_KSearch(10)
            normals = ne.compute()
            normals = normals.to_array()
            matrix = np.zeros((4, 4), dtype=np.float32)
            matrix[0,0] = normals[pc_arrayPos][0]
            matrix[1,0] = normals[pc_arrayPos][1]
            matrix[2,0] = normals[pc_arrayPos][2]
            matrix[3,0] = 0
            matrix[0,1] = math.sqrt(1 + ((normals[pc_arrayPos][0] * normals[pc_arrayPos][0]) / (normals[pc_arrayPos][2] * normals[pc_arrayPos][2])))
            matrix[1,1] = 0
            matrix[2,1] = (math.sqrt(1 + ((normals[pc_arrayPos][0] * normals[pc_arrayPos][0]) / (normals[pc_arrayPos][2] * normals[pc_arrayPos][2])))) * ((-normals[pc_arrayPos][0]) / (normals[pc_arrayPos][2]))
            matrix[3,1] = 0
            matrix[0,2] = 0
            matrix[1,2] = math.sqrt(1 + ((normals[pc_arrayPos][1] * normals[pc_arrayPos][1]) / (normals[pc_arrayPos][2] * normals[pc_arrayPos][2])))
            matrix[2,2] = (math.sqrt(1 + ((normals[pc_arrayPos][1] * normals[pc_arrayPos][1]) / (normals[pc_arrayPos][2] * normals[pc_arrayPos][2])))) * ((-normals[pc_arrayPos][1]) / (normals[pc_arrayPos][2]))
            matrix[0,3] = 0
            matrix[1,3] = 0
            matrix[2,3] = 0
            matrix[3,3] = 1
            r_matrix =  rotation_matrix(angle, (normals[pc_arrayPos]))
            q_matrix = np.dot(matrix, r_matrix)
            q0 = quaternion_from_matrix(q_matrix)
            nor_m = math.sqrt((q0[0]*q0[0] + q0[1]*q0[1] + q0[2]*q0[2] + q0[3]*q0[3]))
            q0 = np.array((q0[0]/nor_m, q0[1]/nor_m, q0[2]/nor_m, q0[3]/nor_m))
            q = quaternion_about_axis(angle, (normals[pc_arrayPos]))
            nor = math.sqrt((q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]))
            q = np.array((q[0]/nor, q[1]/nor, q[2]/nor, q[3]/nor))
            q1 = np.array((0.5,0.5,-0.5,0.5))
            q_f = quaternion_multiply(q, q0)
            q_f = quaternion_multiply(q1, q_f)

            self.pose = PoseStamped()
            self.pose.header.stamp = rospy.Time().now
            self.pose.header.frame_id = 'xtion_rgb_optical_frame'
            self.pose.pose.position.x = gen_list[arrayPosX][0] + (normals[pc_arrayPos][0] * 0.15)
            self.pose.pose.position.y = gen_list[arrayPosY][1] + (normals[pc_arrayPos][1] * 0.15)
            self.pose.pose.position.z = gen_list[arrayPosZ][2] + (normals[pc_arrayPos][2] * 0.15)
            self.pose.pose.orientation.x = 0
            self.pose.pose.orientation.y = 0
            self.pose.pose.orientation.z = 0
            self.pose.pose.orientation.w = 1

            self.pose = self.transform_pose(self.pose)
            self.trans_x = np.append(self.pose.pose.position.x, self.trans_x)
            self.trans_y = np.append(self.pose.pose.position.y, self.trans_y)
            self.trans_z = np.append(self.pose.pose.position.z, self.trans_z)

            self.pose_pre = PoseStamped()
            self.pose_pre.header.stamp = rospy.Time().now
            self.pose_pre.header.frame_id = 'xtion_rgb_optical_frame'
            self.pose_pre.pose.position.x = gen_list[arrayPosX][0] + (normals[pc_arrayPos][0] * 0.25)
            self.pose_pre.pose.position.y = gen_list[arrayPosY][1] + (normals[pc_arrayPos][1] * 0.25)
            self.pose_pre.pose.position.z = gen_list[arrayPosZ][2] + (normals[pc_arrayPos][2] * 0.25)
            self.pose_pre.pose.orientation.x = 0
            self.pose_pre.pose.orientation.y = 0
            self.pose_pre.pose.orientation.z = 0
            self.pose_pre.pose.orientation.w = 1

            self.pose_pre = self.transform_pose(self.pose_pre)
            self.trans_pre_x = np.append(self.pose_pre.pose.position.x, self.trans_pre_x)
            self.trans_pre_y = np.append(self.pose_pre.pose.position.y, self.trans_pre_y)
            self.trans_pre_z = np.append(self.pose_pre.pose.position.z, self.trans_pre_z)

            self.ox = np.append(q_f[0], self.ox)
            self.oy = np.append(q_f[1], self.oy)
            self.oz = np.append(q_f[2], self.oz)
            self.ow = np.append(q_f[3], self.ow)


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
        self.get_object()
        for k in range(len(self.trans_x)):
            self.grasp_pose = PoseStamped()
            self.grasp_pose.header.stamp = rospy.Time().now
            self.grasp_pose.header.frame_id = 'base_footprint'

            self.grasp_pose.pose.position.x = self.trans_pre_x[k]
            self.grasp_pose.pose.position.y = self.trans_pre_y[k]
            self.grasp_pose.pose.position.z = self.trans_pre_z[k]
            self.grasp_pose.pose.orientation.x = self.ox[k]
            self.grasp_pose.pose.orientation.y = self.oy[k]
            self.grasp_pose.pose.orientation.z = self.oz[k]
            self.grasp_pose.pose.orientation.w = self.ow[k]

            if not success:
                self.group.set_pose_target(self.grasp_pose)
                plan=self.group.plan()
                if plan.joint_trajectory.points:
                    print(self.grasp_pose)
                    self.move_to(self.grasp_pose)
                    success=True

            self.grasp_pose.pose.position.x = self.trans_x[k]
            self.grasp_pose.pose.position.y = self.trans_y[k]
            self.grasp_pose.pose.position.z = self.trans_z[k]

            if success:
                print(self.grasp_pose)
                self.move_to_linear(self.grasp_pose)
                break


        self.grip_axcli = actionlib.SimpleActionClient('/play_motion', PlayMotionAction)
        self.grip_axcli.wait_for_server()
        self.grip_goal.motion_name = 'close_gripper'
        self.grip_axcli.send_goal(self.grip_goal)
        self.grip_axcli.wait_for_result()
        rospy.sleep(2)
        self.grasp_pose.pose.position.z += 0.1
        print(self.grasp_pose)
        self.move_to_linear(self.grasp_pose)
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
        rospy.init_node('rcnn_app2')
        call = GraspRGB()
        call.grasp()
