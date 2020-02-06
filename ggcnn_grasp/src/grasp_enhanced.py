#!/usr/bin/env python
# Import all necessary ROS messages and libraries.
import rospy
import copy
import sys
import math

from geometry_msgs.msg import PoseStamped, Quaternion, TransformStamped
from moveit_commander import RobotCommander, PlanningSceneInterface
from moveit_msgs.msg import (Constraints, MoveItErrorCodes,
                             OrientationConstraint, PickupAction, PickupGoal,
                             PlaceAction, PlaceGoal, PlanningOptions, PickupResult)
from moveit_msgs.msg import Grasp
import moveit_commander
from actionlib import SimpleActionClient, SimpleActionServer
from std_msgs.msg import Float32MultiArray
import numpy as np
from tf import transformations as tft
import tf
import tf2_ros
import tf2_geometry_msgs
from gazebo_msgs.srv import GetModelStateRequest, GetModelState
from trajectory_msgs.msg import JointTrajectoryPoint
from iki_manipulation.recognized_object import RecognizedObject
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import pcl
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from tf.transformations import quaternion_about_axis, rotation_matrix, quaternion_from_matrix, quaternion_multiply
import actionlib
from play_motion_msgs.msg import PlayMotionAction, PlayMotionGoal
bridge = CvBridge()

class GraspIt:
    def __init__(self):
        # Initialize node.
        rospy.init_node('grasp_app2')
        # Initialize MoveIt.
        self.robot = RobotCommander()
        self.scene = PlanningSceneInterface()
        self.arm = self.robot.get_group('arm_torso')
        self.group = moveit_commander.MoveGroupCommander("arm_torso")
        self.gripper = self.robot.get_group('gripper')
        self.arm.set_planning_time(0.0)
        self.arm.set_planner_id("RRTConnectkConfigDefault")
        self.arm.set_pose_reference_frame("base_footprint")
        self.arm.allow_replanning(False)
        # Initialize required variables and assign them to specific message types.
        self.grasp_msg = Image()
        self.angle_msg = Image()
        self.pcd_message = PointCloud2()
        self.target_grasp = PoseStamped()
        self.tmpPoseStamped = PoseStamped()
        self.mid = PoseStamped()
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
        self.grip_goal = PlayMotionGoal()
        self.scene_object = RecognizedObject("part")
        # Initialize TF transformations.
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(1200.0))

    def get_grasp(self):
        # Function to calculate grasps and execute a grasp.
        eef_step = 0.01
        jump_threshold = 0.0
        # Get grasp map, angle map, and point cloud.
        self.grasp_msg = rospy.wait_for_message('/ggcnn/img/points', Image)
        self.angle_msg = rospy.wait_for_message('/ggcnn/img/ang', Image)
        self.pcd_message = rospy.wait_for_message('/xtion/depth_registered/points', PointCloud2)

        gen = pc2.read_points(self.pcd_message, skip_nans=False, field_names=("x", "y", "z"))
        gen_list = list(gen)
        pc = pc2.read_points(self.pcd_message, skip_nans=True, field_names=("x", "y", "z"))
        pc_list = []
        for p in pc:
            pc_list.append( [p[0],p[1],p[2]] )
        # Convert ROS images to CV2.
        grasp_img = bridge.imgmsg_to_cv2(self.grasp_msg)
        ang_msg = bridge.imgmsg_to_cv2(self.angle_msg)
        # Get the 50 best grasps.
        maxes = self.largest_indices(grasp_img, 50)
        # Calculate orientation of all grasps using surface normal and angle.
        for i in range(len(maxes[0])):
            angle = abs(ang_msg[maxes[0][i]][maxes[1][i]]) + np.pi/2
            pred_x = ((maxes[1][i] * 400) / 300) + 120
            pred_y = ((maxes[0][i] * 400) / 300) + 40
            arrayPos = pred_y*self.pcd_message.width + pred_x
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
            q = quaternion_about_axis(angle, (normals[pc_arrayPos])) # Needs to be normalized
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

            self.pose = self.transform_grasp(self.pose)
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

            self.pose_pre = self.transform_grasp(self.pose_pre)
            self.trans_pre_x = np.append(self.pose_pre.pose.position.x, self.trans_pre_x)
            self.trans_pre_y = np.append(self.pose_pre.pose.position.y, self.trans_pre_y)
            self.trans_pre_z = np.append(self.pose_pre.pose.position.z, self.trans_pre_z)

            self.ox = np.append(q_f[0], self.ox)
            self.oy = np.append(q_f[1], self.oy)
            self.oz = np.append(q_f[2], self.oz)
            self.ow = np.append(q_f[3], self.ow)


        for k in range(len(self.trans_x)):
            # Execute grasps.
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

            success=False
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

        # Close Gripper.
        self.grip_axcli = actionlib.SimpleActionClient('/play_motion', PlayMotionAction)
        self.grip_axcli.wait_for_server()
        self.grip_goal.motion_name = 'close_gripper'
        self.grip_axcli.send_goal(self.grip_goal)
        self.grip_axcli.wait_for_result()
        rospy.sleep(2)
        # Post Grasp.
        self.target_grasp.pose.position.z += 0.10
        print(self.target_grasp)
        self.move_to_linear(self.target_grasp)

    def get_object(self):
        # Use gazebo/get_model_state service to get object position.
        rospy.wait_for_service('/gazebo/get_model_state')
        self.target_service = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.target_request = GetModelStateRequest()
        # Specify name of the object to get position.
        self.target_request.model_name = 'cola_big'
        self.target_request.relative_entity_name = 'base_footprint'
        self.target_result = self.target_service(self.target_request)
        tmpPoseStamped = PoseStamped()
        tmpPoseStamped.header.frame_id = self.target_result.header.frame_id
        tmpPoseStamped.header.stamp = rospy.Time(0)
        tmpPoseStamped.pose = self.target_result.pose
        self.scene_object.name = self.target_request.model_name
        self.scene_object.pose = tmpPoseStamped
        # Give object dimensions to add to planning scene.
        self.scene_object.dimensions = (0.08, 0.08, 0.3)
        self.add_pickup_object(self.scene_object)

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

        target_pose_stamped_base_link = self._transform_pose_to_frame(target_pose_stamped, "base_footprint")

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

    def transform_grasp(self, target):
        # Function to transform grasp from camera frame to base frame.
        listener = tf.TransformListener()
        try:
            listener.waitForTransform('/base_footprint', '/xtion_rgb_optical_frame', rospy.Time(0), rospy.Duration(5))
            (trans, rot) = listener.lookupTransform('/base_footprint', '/xtion_rgb_optical_frame', rospy.Time(0))
        except Exception as e:
            rospy.logerr('#'*20+"\n\n"+str(e))
            return False
        p = PoseStamped()
        p.header.stamp=rospy.Time()
        p.header.frame_id='xtion_rgb_optical_frame'
        p.pose.position = target.pose.position
        p.pose.orientation = target.pose.orientation
        t = TransformStamped()
        t.transform.translation.x = trans[0]
        t.transform.translation.y = trans[1]
        t.transform.translation.z = trans[2]
        t.transform.rotation.x = rot[0]
        t.transform.rotation.y = rot[1]
        t.transform.rotation.z = rot[2]
        t.transform.rotation.w = rot[3]
        h=tf2_geometry_msgs.do_transform_pose(p,t)
        h.header.frame_id = '/base_footprint'
        h.header.stamp = rospy.Time()
        return h

    def largest_indices(self, ary, n):
        # Function to get the indices of n largest values in an array.
        flat = ary.flatten()
        indices = np.argpartition(flat, -n)[-n:]
        indices = indices[np.argsort(-flat[indices])]
        return np.unravel_index(indices, ary.shape)

    def list_to_quaternion(self, l):
        # Function to convert a list to quaternion.
        q = Quaternion()
        q.x = l[0]
        q.y = l[1]
        q.z = l[2]
        q.w = l[3]
        return q

    def move_to(self, pose_stamped):
        # Function to move arm to a pose in the cartesian space.
        self.arm.set_start_state_to_current_state()
        self.arm.set_pose_target(pose_stamped)
        result = self.arm.go(wait=True)
        return result

    def add_pickup_object(self, recognized_object):
        # Function to add an object to the planning scene.
        pose_stamped = copy.deepcopy(recognized_object.pose)
        self.scene.add_box(recognized_object.name, pose_stamped,
                           (recognized_object.dimensions[0], recognized_object.dimensions[1],
                            recognized_object.dimensions[2]))

        rospy.loginfo("Object {} added to planning scene".format(recognized_object.name))

        rospy.logdebug("Object {} added with pose \n{} \nand dimensions {}".format(recognized_object.name, pose_stamped,
                                                                                   recognized_object.dimensions))

if __name__ == '__main__':
    g = GraspIt()
    g.get_object()
    g.get_grasp()
