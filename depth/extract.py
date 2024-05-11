from rosbags.rosbag2 import Reader
from collections import defaultdict
import numpy as np

# Path to the ROS bag directory
bag_path = '/Volumes/T7 Shield/hawaii/'

# List of topics to analyze
topics_to_analyze = [
    '/novatel_bottom/imu/raw_data',
]

# Initialize dictionary to hold timestamps for each topic
timestamps = defaultdict(list)

with Reader(bag_path) as reader:
    # Iterate through all messages in the reader
    for connection, timestamp, rawdata in reader.messages():
        if connection.topic in topics_to_analyze:
            # Convert timestamp from nanoseconds to seconds
            timestamps[connection.topic].append(timestamp * 1e-9)

# Analyze the frequency for each topic
frequencies = {}
for topic, times in timestamps.items():
    if len(times) > 1:
        # Calculate time differences between consecutive messages
        intervals = np.diff(times)
        # Calculate average frequency
        mean_interval = np.mean(intervals)
        frequency = 1.0 / mean_interval if mean_interval > 0 else 0
        frequencies[topic] = frequency
    else:
        frequencies[topic] = 0  # No frequency calculation possible with one timestamp

    #print novatel_bottom_imu)

# Determine the maximum length of topic names
max_length = max(len(topic) for topic in frequencies.keys())

# Output the frequencies with padded topic names
for topic, freq in sorted(frequencies.items()):
    print(f"{topic.ljust(max_length)} : {freq:.2f} Hz")

