#!/usr/bin/env python
# Import all necessary ROS messages and libraries.
import rospy
import copy
import sys

from geometry_msgs.msg import PoseStamped, Quaternion, TransformStamped
from moveit_commander import RobotCommander, PlanningSceneInterface
from moveit_msgs.msg import (Constraints, MoveItErrorCodes,
                             OrientationConstraint, PickupAction, PickupGoal,
                             PlaceAction, PlaceGoal, PlanningOptions, PickupResult)
from moveit_msgs.msg import Grasp
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
import actionlib
from play_motion_msgs.msg import PlayMotionAction, PlayMotionGoal


class GraspIt:
    def __init__(self):
        # Initialize node.
        rospy.init_node('grasp_app1')
        # Initialize MoveIt.
        self.robot = RobotCommander()
        self.scene = PlanningSceneInterface()
        self.arm = self.robot.get_group('arm_torso')
        self.gripper = self.robot.get_group('gripper')
        self.arm.set_planning_time(0.0)
        self.arm.set_planner_id("RRTConnectkConfigDefault")
        self.arm.set_pose_reference_frame("base_footprint")
        self.arm.allow_replanning(False)
        # Initialize required variables and assign them to specific message types.
        self.grasp_det_msg = Float32MultiArray()
        self.target_grasp = PoseStamped()
        self.tmpPoseStamped = PoseStamped()
        self.scene_object = RecognizedObject("part")
        self.grip_goal = PlayMotionGoal()
        # Initialize TF transformations.
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(1200.0))

    def get_grasp(self):
        # Function to recieve and execute grasp from the GGCNN.
        eef_step = 0.01
        jump_threshold = 0.0
        self.grasp_det_msg = rospy.wait_for_message('/ggcnn/out/command', Float32MultiArray)
        self.target_grasp.header.frame_id = '/xtion_rgb_optical_frame'
        self.target_grasp.header.stamp = rospy.Time.now()
        self.target_grasp.pose.position.x = self.grasp_det_msg.data[0]
        self.target_grasp.pose.position.y = self.grasp_det_msg.data[1]
        self.target_grasp.pose.position.z = self.grasp_det_msg.data[2]
        self.target_grasp.pose.orientation.w = 1.0
        self.target_grasp = self.transform_grasp(self.target_grasp)
        if self.scene_object.dimensions[2] > 0.14:
            # Orientation for front grasps.
            self.target_grasp.pose.orientation = self.list_to_quaternion(tft.quaternion_from_euler(np.pi/2 + self.grasp_det_msg.data[3], 0, 0))
            # Pre grasp for front grasps.
            self.target_grasp.pose.position.x -= 0.25
            print self.target_grasp
            self.move_to(self.target_grasp)
            # Grasp for front grasps.
            self.target_grasp.pose.position.x += 0.10
            print(self.target_grasp)
            self.move_to_linear(self.target_grasp)
            # Close Gripper.
            self.grip_axcli = actionlib.SimpleActionClient('/play_motion', PlayMotionAction)
            self.grip_axcli.wait_for_server()
            self.grip_goal.motion_name = 'close_gripper'
            self.grip_axcli.send_goal(self.grip_goal)
            self.grip_axcli.wait_for_result()
            rospy.sleep(2)
            # Post Grasp for Front grasps.
            self.target_grasp.pose.position.z += 0.10
            print(self.target_grasp)
            self.move_to_linear(self.target_grasp)
        else:
            # Orientation for top down grasps. Need to add angle param.
            self.target_grasp.pose.orientation = self.list_to_quaternion(tft.quaternion_from_euler(np.pi/2, np.pi/2, self.grasp_det_msg.data[3]))
            # Pre-Grasp for top down grasps.
            self.target_grasp.pose.position.z += 0.30
            print self.target_grasp
            self.move_to(self.target_grasp)
            # Grasp for top down grasps
            self.target_grasp.pose.position.z -= 0.15
            print(self.target_grasp)
            self.move_to_linear(self.target_grasp)
            # Close Gripper.
            self.grip_axcli = actionlib.SimpleActionClient('/play_motion', PlayMotionAction)
            self.grip_axcli.wait_for_server()
            self.grip_goal.motion_name = 'close_gripper'
            self.grip_axcli.send_goal(self.grip_goal)
            self.grip_axcli.wait_for_result()
            rospy.sleep(2)
            # Post Grasp for top down grasps.
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
