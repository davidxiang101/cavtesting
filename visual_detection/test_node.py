import cv2
import argparse
import numpy as np
import sensor_msgs.msg
import apriltag

def receive_image(compressed_image_msg: sensor_msgs.msg.CompressedImage):
    try:
        # Convert ROS Image message to OpenCV image
        compressed_image_buffer = np.frombuffer(compressed_image_msg.data, dtype=np.uint8)
        image_np = cv2.imdecode(compressed_image_buffer, cv2.IMREAD_COLOR)

        # Convert image to grayscale
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (15, 15), 0)

        # AprilTag detection
        detector = apriltag.Detector(apriltag.DetectorOptions(families="tag36h11"))
        results = detector.detect(gray)
        print(f"{len(results)} total AprilTags detected")
        
        tag_data = {}  # Initialize an empty dictionary to store tag IDs and corners    

        # Process each detection
        for tag in results:
            # Extract corners and draw bounding box
            (ptA, ptB, ptC, ptD) = tag.corners

            tag_data[tag.tag_id] = [(int(pt[0]), int(pt[1])) for pt in tag.corners]

            points = [(int(pt[0]), int(pt[1])) for pt in [ptA, ptB, ptC, ptD]]
            cv2.polylines(image_np, [np.array(points)], isClosed=True, color=(0, 255, 0), thickness=2)

            # Draw center point
            center = (int(tag.center[0]), int(tag.center[1]))
            cv2.circle(image_np, center, 5, (0, 0, 255), -1)

            # Tag information
            tag_family = tag.tag_family.decode('utf-8')
            cv2.putText(image_np, tag_family, (points[0][0], points[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the result
        cv2.imshow("AprilTag Detection", image_np)
        cv2.waitKey(0)
    except Exception as e:
        print("failed to process image: ", str(e))

    # return all corners
    return tag_data

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image containing AprilTag")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])

# Encode the image to JPEG format
result, encimg = cv2.imencode('.jpg', image)
if not result:
    raise Exception("Image encoding failed")

# Create a CompressedImage message
compressed_image_msg = sensor_msgs.msg.CompressedImage()
compressed_image_msg.format = "jpeg"
compressed_image_msg.data = np.array(encimg).tobytes()

corners = receive_image(compressed_image_msg)

print(corners)