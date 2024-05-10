import numpy as np

import cv2
import PIL.Image

from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray
from geometry_msgs.msg import PolygonStamped, Polygon, Point32
import sensor_msgs.msg

# Locate a plane from a point cloud.
import open3d as o3d
import open3d

import apriltag

class CalibrationNode(Node):
    """Class object central for trajectory prediction."""

    def __init__(self):
        super().__init__("visual_detection_node")

        self.camera_id = self.declare_parameter(
            name="camera_id",
            value=None,
        ).value

        self.detector = apriltag.Detector(apriltag.DetectorOptions(families="tag36h11"))

        share_directory = get_package_share_directory("visual_detection")

        history_depth = 10
        self.image_subscription = self.create_subscription(
            msg_type=sensor_msgs.msg.CompressedImage,
            topic="image",
            callback=self.receive_image,
            qos_profile=history_depth,
        )
        self.pcd_subscription = self.create_subscription(
            msg_type=sensor_msgs.msg.PointCloud2,

            topic="point_cloud",
            callback=self.receive_point_cloud,
            qos_profile=history_depth,
        )

        self.get_logger().info("Node started!")

    def receive_point_cloud(self, point_cloud_msg: sensor_msgs.msg.PointCloud2):
        # Create an Open3D point cloud from the ROS message
        # ...
        pcd_point_cloud = o3d.data.PCDPointCloud()
        pcd = o3d.io.read_point_cloud(pcd_point_cloud.path)

        plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                                ransac_n=3,
                                                num_iterations=1000)
        [a, b, c, d] = plane_model
        print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

        inlier_cloud = pcd.select_by_index(inliers)
        inlier_cloud.paint_uniform_color([1.0, 0, 0])
        outlier_cloud = pcd.select_by_index(inliers, invert=True)
        o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                        zoom=0.8,
                                        front=[-0.4999, -0.1659, -0.8499],
                                        lookat=[2.1813, 2.0619, 2.0999],
                                        up=[0.1204, -0.9852, 0.1215])
        pass

    def receive_image(self, compressed_image_msg: sensor_msgs.msg.CompressedImage):
        # Can be found in /camera/info/topic/name/camera_info
        # Specifically, the fields roi.height and roi.weight
        image_height = 960
        image_width = 2064

        tag_data = {}  # Initialize an empty dictionary to store tag IDs and corners

        try:
            # Convert ROS Image message to OpenCV image
            compressed_image_buffer = np.frombuffer(compressed_image_msg.data, dtype=np.uint8)
            image_np = cv2.imdecode(compressed_image_buffer, cv2.IMREAD_COLOR)

            # Convert image to grayscale
            gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (15, 15), 0)

            # AprilTag detection
            results = self.detector.detect(gray)
            self.get_logger().info(f"{len(results)} total AprilTags detected")

            for tag in results:
                corners = [(int(pt[0]), int(pt[1])) for pt in tag.corners]
                tag_data[tag.tag_id] = corners

            # # Process each detection
            # for tag in results:
            #     # Extract corners and draw bounding box
            #     (ptA, ptB, ptC, ptD) = tag.corners
            #     points = [(int(pt[0]), int(pt[1])) for pt in [ptA, ptB, ptC, ptD]]
            #     cv2.polylines(image_np, [np.array(points)], isClosed=True, color=(0, 255, 0), thickness=2)

            #     # Draw center point
            #     center = (int(tag.center[0]), int(tag.center[1]))
            #     cv2.circle(image_np, center, 5, (0, 0, 255), -1)

            #     # Tag information
            #     tag_family = tag.tag_family.decode('utf-8')
            #     cv2.putText(image_np, tag_family, (points[0][0], points[0][1] - 10),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display the result
            # cv2.imshow("AprilTag Detection", image_np)
            # cv2.waitKey(0)
        except Exception as e:
            self.get_logger().error(f"Failed to process image: {str(e)}")

        # return all corners
        return tag_data

        
    def shutdown_ros(self, *args):
        # Handle shutdown.
        pass


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

    calibration_node = CalibrationNode()

    rclpy.spin(calibration_node)

    signal.signal(signal.SIGINT, calibration_node.shutdown_ros)

    rclpy.shutdown()

    calibration_node.destroy_node()


if __name__ == "__main__":
    main()
