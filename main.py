import os
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO
import seaborn as sns

# Install torch using pip from within the script
os.system('pip install torch')

# Now you can import torch and use it in your script
import torch

from tracker import Tracker

# Check if a GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print("GPU is available and will be utilized.")
else:
    print("GPU is not available. Using CPU.")

video_path = os.path.join('.', 'data', 'FinalData_480p(2x).mp4')
video_out_path = os.path.join('.', 'output_data.mp4')

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS),
                          (frame.shape[1], frame.shape[0]))

model = YOLO("yolov8n.pt").to(device)

tracker = Tracker()

colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

# Use Seaborn color palette for line plots
line_plot_colors = sns.color_palette("husl", 10)

# Data storage for trajectories and speeds
trajectory_data = {}
speed_data = {}

detection_threshold = 0.05
while ret:

    results = model(frame)

    for result in results:
        detections = []
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            class_id = int(class_id)
            if score > detection_threshold:
                detections.append([x1, y1, x2, y2, score])

        tracker.update(frame, detections)

        for track in tracker.tracks:
            bbox = track.bbox
            x1, y1, x2, y2 = bbox
            track_id = track.track_id

            # Append the current position to the trajectory data
            if track_id not in trajectory_data:
                trajectory_data[track_id] = []
            trajectory_data[track_id].append((int((x1 + x2) / 2), int((y1 + y2) / 2)))

            # Calculate speed (assuming constant frame rate)
            if len(trajectory_data[track_id]) > 1:
                last_position = np.array(trajectory_data[track_id][-2])
                current_position = np.array(trajectory_data[track_id][-1])
                speed = np.linalg.norm(current_position - last_position)

                # Append the speed to the speed data
                if track_id not in speed_data:
                    speed_data[track_id] = []
                speed_data[track_id].append(speed)


            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)

    cap_out.write(frame)
    ret, frame = cap.read()

cap.release()
cap_out.release()
cv2.destroyAllWindows()

# Display all trajectories in one graph
plt.figure()
for track_id, trajectory in trajectory_data.items():
    plt.plot([pos[0] for pos in trajectory], [pos[1] for pos in trajectory], label=f'Track {track_id}', color=line_plot_colors[track_id % len(line_plot_colors)])

plt.title('All Trajectories')
plt.xlabel('X Position (in pixels)')
plt.ylabel('Y Position (in pixels)')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, ncol=len(trajectory_data))

# Save the trajectory graph as a JPEG file
plt.savefig('trajectory_graph.jpg')

# Display all speeds in another graph
plt.figure()
for track_id, speeds in speed_data.items():
    plt.plot(speeds, label=f'Track {track_id}', color=line_plot_colors[track_id % len(line_plot_colors)])

plt.title('All Speeds')
plt.xlabel('Frame Number')
plt.ylabel('Speed (in pixels/frame)')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, ncol=len(speed_data))

# Save the speed graph as a JPEG file
plt.savefig('speed_graph.jpg')

plt.show()