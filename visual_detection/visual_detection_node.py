import json
import os
import numpy as np

import cv2
import PIL.Image
import torch
import yolov5
from yolov5.models.common import Detections

from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray
from geometry_msgs.msg import PolygonStamped, Polygon, Point32
import sensor_msgs.msg

class VisualDetectionNode(Node):
    """Class object central for trajectory prediction."""

    def __init__(self):
        super().__init__("visual_detection_node")

        self.use_cuda = self.declare_parameter(
            name="use_cuda",
            value=False,
        ).value
        self.camera_id = self.declare_parameter(
            name="camera_id",
            value=None,
        ).value

        # Load camera extrinsics, intrinsics, and distortion coefficients
        # (These will be in keys `ext`, `int`, and `[i forgor]` respectively)
        share_directory = get_package_share_directory("visual_detection")
        with open(os.path.join(share_directory, "cameras/all.json"), "r") as f:
            all_camera_info = json.load(f)
        self.camera_info = all_camera_info[self.camera_id]

        share_directory = get_package_share_directory("visual_detection")

        self.model = yolov5.YOLOv5(os.path.join(share_directory, "models/last.pt"))

        history_depth = 10
        self.predicted_objects_publisher = self.create_publisher(
            msg_type=Int32MultiArray,
            topic="bounding_boxes",
            qos_profile=history_depth,
        )

        self.predicted_objects_3d_publisher = self.create_publisher(
            msg_type=PolygonStamped,
            topic="bounding_boxes_3d",
            qos_profile=history_depth,
        )

        self.fov_publisher = self.create_publisher(
            msg_type=PolygonStamped,
            topic="fov",
            qos_profile=history_depth,
        )

        inverted_extrinsics = invert_extrinsics(np.array(self.camera_info["ext"]))
        self.get_logger().info("Inverted extrinsics: " + repr(inverted_extrinsics))

        self.image_subscription = self.create_subscription(
            msg_type=sensor_msgs.msg.CompressedImage,
            topic="image",
            callback=self.receive_image,
            qos_profile=history_depth,
        )

        self.get_logger().info("Node started!")

    def receive_image(self, compressed_image_msg: sensor_msgs.msg.CompressedImage):
        # Can be found in /camera/info/topic/name/camera_info
        # Specifically, the fields roi.height and roi.weight
        image_height = 960
        image_width = 2064

        compressed_image_buffer = np.array(compressed_image_msg.data)
        image_np = cv2.imdecode(compressed_image_buffer, cv2.IMREAD_COLOR)

        image_pil = PIL.Image.fromarray(image_np)

        with torch.no_grad():
            detections: Detections = self.model.predict([image_pil])
            # should look into how `Detections` is actually structured
            # detections.xywh[0] corresponds to class ID 0.
            for detection in detections.xywh[0]:
                # self.get_logger().info("Deetection: " + repr(detection))
                # (x, y) is center
                # so the corners are offset back by 1/2 <w, h>
                x = int(detection[0] - detection[2] / 2)
                y = int(detection[1] - detection[3] / 2)
                w = int(detection[2])
                h = int(detection[3])

                if self.camera_id == 'front_right':
                    if x > y:
                        continue

                self.predicted_objects_publisher.publish(Int32MultiArray(
                    data=[x, y, x + w, y + h]
                ))
                cv2.rectangle(image_np, (x, y), (x + w, y + h), (0, 255, 0), 5)

                ### Project our 2D bounding box into 3D space ###
            ### OUTSIDE OF LOOP ###
            # But instead, project our image boundary into 3D space
            # x = 0
            # y = 0
            # w = 2064
            # h = 960
                (ox, oy, oz) = project_points_from_camera(
                    np.array(self.camera_info["int"]),
                    np.array(self.camera_info["ext"])[:3, :],
                    np.array([[0, 0]]),
                    z=0
                )[0]
                origin = Point32(x=ox, y=oy, z=oz)
                # Assume that height = C / z (C is some arbitrary constant; probably should be based on some calculations, but I know C is a constant so
                # finding a good value of C through experimentation will be functionally the same as calculating it manually)
                z = 4000 / h
                # Draw a sort of bounding box, with rays coming out from the camera location
                bounding_box_3d_points = [
                    Point32(x=px, y=py, z=pz)
                    for (px, py, pz) in
                    project_points_from_camera(
                        np.array(self.camera_info["int"]),
                        np.array(self.camera_info["ext"])[:3, :],
                        np.array([
                            [2064-x, 960-y],
                            [2064-(x + w), 960-y],
                            [2064-(x + w), 960-(y + h)],
                            [2064-x, 960-(y + h)],
                        ]),
                        z=z
                    )
                ]
                polygon_points = [
                    origin,
                    *bounding_box_3d_points,
                    bounding_box_3d_points[0],
                    bounding_box_3d_points[3],
                    origin,
                    bounding_box_3d_points[2],
                    bounding_box_3d_points[1],
                ]
                projection_3d = PolygonStamped()
                # This defines what coordinate system we are relative to
                projection_3d.header.frame_id = "rear_axle_middle"
                # This is the time the message was generated
                projection_3d.header.stamp = self.get_clock().now().to_msg()
                polygon = Polygon()
                polygon.points = polygon_points
                # self.get_logger().info("Projected 3D points: " + repr(polygon.points))
                projection_3d.polygon = polygon
                self.predicted_objects_3d_publisher.publish(projection_3d)

            ### Display the FOV of the camera ###
            x = 0
            y = 0
            w = 2064
            h = 960
            (ox, oy, oz) = project_points_from_camera(
                np.array(self.camera_info["int"]),
                np.array(self.camera_info["ext"])[:3, :],
                np.array([[0, 0]]),
                z=0
            )[0]
            origin = Point32(x=ox, y=oy, z=oz)
            # Draw a sort of bounding box, with rays coming out from the camera location
            bounding_box_3d_points = [
                Point32(x=px, y=py, z=pz)
                for (px, py, pz) in
                project_points_from_camera(
                    np.array(self.camera_info["int"]),
                    np.array(self.camera_info["ext"])[:3, :],
                    np.array([
                        [2064-x, 960-y],
                        [2064-(x + w), 960-y],
                        [2064-(x + w), 960-(y + h)],
                        [2064-x, 960-(y + h)],
                    ]),
                    z=2
                )
            ]
            polygon_points = [
                origin,
                *bounding_box_3d_points,
                bounding_box_3d_points[0],
                bounding_box_3d_points[3],
                origin,
                bounding_box_3d_points[2],
                bounding_box_3d_points[1],
            ]
            fov_polygon_message = PolygonStamped()
            # This defines what coordinate system we are relative to
            fov_polygon_message.header.frame_id = "rear_axle_middle"
            # This is the time the message was generated
            fov_polygon_message.header.stamp = self.get_clock().now().to_msg()
            polygon = Polygon()
            polygon.points = polygon_points
            # self.get_logger().info("Projected 3D points: " + repr(polygon.points))
            fov_polygon_message.polygon = polygon
            self.fov_publisher.publish(fov_polygon_message)

        # Plot in BGR, because this is how OpenCV works
        if self.camera_id == 'front_left_center':
            cv2.imshow("Received image", image_np[:, :, ::-1])
            cv2.waitKey(1)
        

    def shutdown_ros(self, *args):
        # Handle shutdown.
        pass

def invert_extrinsics(extrinsics):

    """
    The extrinsic matrix is 3x4. Therefore, we can't simply take a matrix inverse.
    We need to inver the rotation matrix and translation matrix separately.
    Specifically, we:
    (1) invert the rotation matrix, and
    (2) multiply the translation matrix by the inverse rotation matrix, and
    (3) invert the translation by negating the translation matrix
    """
    # Get 3x3 rotation matrix
    R = extrinsics[:3, :3]
    # Get 3x1 translation matrix
    C = np.expand_dims(extrinsics[:3, 3], 0).T
    # inverse of rotation matrix is transpose, as dimensions are orthonormal.
    # Invert C
    R_inv = R.T
    R_inv_C = np.matmul(R_inv, C)
    # Create new matrix
    result = np.concatenate((R_inv, -R_inv_C), -1)
    result = np.concatenate((result, np.zeros((1, 4))), 0)
    result[3, :3] = 0
    result[3, 3] = 1
    return result

def project_points_from_camera(intrinsic: np.ndarray, extrinsic: np.ndarray, points: np.ndarray, z: float):
    points_in_pixel_space = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    """
    Scale by 3 to extend forward by 3 meters instead.
    """
    points_in_pixel_space = points_in_pixel_space * z
    """
    To convert these "pixel space" points into "camera frame", we recall
    that the intrinsic matrix is what took us from "camera frame" to "pixel
    space". So, we just invert the intrinsic matrix.
    """
    intrinsic_matrix_inverse = np.linalg.inv(intrinsic)
    points_in_camera_frame = np.matmul(intrinsic_matrix_inverse, points_in_pixel_space.T).T
    """
    We now have [x y z] points. To use the extrinsic matrix to go from camera_frame -> rear_axle_middle_ground frame,
    we must add an extra dimension at the end to make the points [x y z 1]. This extra dimension is necessary simply
    because we make a translation, and the 1 is used to create coefficients (recall tx, ty, tz below).
    """
    points_in_camera_frame = np.concatenate((points_in_camera_frame, np.ones((points_in_camera_frame.shape[0], 1))), axis=1)
    points_in_world_frame = np.matmul(invert_extrinsics(extrinsic), points_in_camera_frame.T).T
    points_in_world_frame = points_in_world_frame[:, :3]

    return points_in_world_frame

def main(args=None):
    import signal

    """Execute mix_net node."""
    rclpy.init(args=args)

    visual_detection_node = VisualDetectionNode()

    rclpy.spin(visual_detection_node)

    signal.signal(signal.SIGINT, visual_detection_node.shutdown_ros)

    rclpy.shutdown()

    visual_detection_node.destroy_node()


if __name__ == "__main__":
    main()
