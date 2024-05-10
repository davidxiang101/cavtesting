import os

import cv2

from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray
from nav_msgs.msg import Odometry
from functools import partial
import json
import numpy as np

class TriangulationNode(Node):
    """Class object central for trajectory prediction."""

    def __init__(self):
        super().__init__("triangulation_node")

        self.use_cuda = self.declare_parameter(
            name="use_cuda",
            value=False,
        ).value

        share_directory = get_package_share_directory("visual_detection")

        with open(os.path.join(share_directory, "cameras/all.json"), "r") as f:
            cams = json.load(f)

        self.camera1_info = cams['front_left_center']
        self.camera2_info = cams['front_right']

        history_depth = 10
        self.predicted_objects_publisher = self.create_publisher(
            msg_type=Int32MultiArray,
            topic="bounding_boxes",
            qos_profile=history_depth,
        )

        self.camera1_detections = self.create_subscription(
            msg_type=Int32MultiArray,
            topic="camera1_detections",
            callback=partial(self.receive_detection, "cam1"),
            qos_profile=history_depth,
        )
        self.camera2_detections = self.create_subscription(
            msg_type=Int32MultiArray,
            topic="camera2_detections",
            callback=partial(self.receive_detection, "cam2"),
            qos_profile=history_depth,
        )
        self.triangulated_objects_publisher = self.create_publisher(
            msg_type=Odometry,
            topic="triangulated_points",
            qos_profile=history_depth,
        )

        self.prev_detections = {}

        self.get_logger().info("Node started!")

    def receive_detection(self, camera_id: str, bounding_box: Int32MultiArray):
        self.prev_detections[camera_id] = list(bounding_box.data)

        if len(self.prev_detections) < 2:
            return
        
        x11, y11, x21, y21 = self.prev_detections['cam1']
        x12, y12, x22, y22 = self.prev_detections['cam2']

        center_x1 = (x11 + x21) / 2
        center_x2 = (x12 + x22) / 2
        center_y1 = (y11 + y21) / 2
        center_y2 = (y12 + y22) / 2

        camera1_extrinsics = np.array(self.camera1_info['ext'])[:3, :]
        camera2_extrinsics = np.array(self.camera2_info['ext'])[:3, :]
        camera1_intrinsics = np.array(self.camera1_info['int'])
        camera2_intrinsics = np.array(self.camera2_info['int'])
        c1d = np.array(self.camera1_info['dis'])
        c2d = np.array(self.camera2_info['dis'])

        pt1 = cv2.undistortPoints(np.array([center_x1, center_y1]), camera1_intrinsics, c1d)
        pt2 = cv2.undistortPoints(np.array([center_x2, center_y2]), camera2_intrinsics, c2d)

        triangulated_homogenous = cv2.triangulatePoints(camera1_extrinsics, camera2_extrinsics, pt1, pt2)
        triangulated_homogenous = triangulated_homogenous.T
        # Divide xyz by homogenous points
        triangulated = triangulated_homogenous[:, :3] / triangulated_homogenous[:, -1]
        triangulated = triangulated[0]
        # self.get_logger().info("Triangulated: " + repr(triangulated))

        odometry = Odometry()
        odometry.header.frame_id = "rear_axle_middle_ground"
        odometry.pose.pose.position.x, odometry.pose.pose.position.y, odometry.pose.pose.position.z = triangulated

        self.triangulated_objects_publisher.publish(odometry)
        
        self.prev_detections.clear()

    def shutdown_ros(self, *args):
        # Handle shutdown.
        pass

def main(args=None):
    import signal

    rclpy.init(args=args)

    node = TriangulationNode()

    rclpy.spin(node)

    signal.signal(signal.SIGINT, node.shutdown_ros)

    rclpy.shutdown()

    node.destroy_node()


if __name__ == "__main__":
    main()
