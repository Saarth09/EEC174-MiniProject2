import cv2
import numpy as np
from tracker import Tracker


# Load YOLO model
net = cv2.dnn.readNetFromDarknet("yolo_files/yolov3.cfg", "yolo_files/yolov3.weights")
layer_names = net.getLayerNames()
try:
    yolo_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
except (IndexError, TypeError):
    yolo_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load COCO class labels (track people only)
with open("yolo_files/coco.names", "r") as f:
    class_names = [line.strip() for line in f.readlines()]
PERSON_CLASS_ID = class_names.index("person")

CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.3

# Load video
video_path = "input.webm"
cap = cv2.VideoCapture(video_path)

# Initialize the tracker
tracker = Tracker(maxLost=30)

# Determine the total number of frames in the video
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Set up the video writer for output
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # For MP4 format
out_vid = cv2.VideoWriter("output.mp4", fourcc, fps, (frame_w, frame_h))

# Consistent colors per ID
np.random.seed(42)
colors = np.random.randint(0, 255, size=(200, 3), dtype="uint8")


def draw_paths(frame, paths, max_len=30):
    """Draw centroid trails for each tracked ID using cv2.line()."""
    for objectID, path in paths.items():
        if len(path) < 2:
            continue
        trail = path[-max_len:]
        color = [int(c) for c in colors[int(objectID) % len(colors)]]
        for i in range(1, len(trail)):
            pt1 = (int(trail[i - 1][0]), int(trail[i - 1][1]))
            pt2 = (int(trail[i][0]), int(trail[i][1]))
            # Fade older segments slightly by thickness
            thickness = max(1, int(2 * i / len(trail)))
            cv2.line(frame, pt1, pt2, color, thickness)


# Process each frame
for _ in range(total_frames):
    ret, frame = cap.read()
    if not ret:
        break

    (H, W) = frame.shape[:2]

    # Convert frame to blob
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Use YOLO net to detect objects
    layer_outputs = net.forward(yolo_layers)

    boxes = []
    confidences = []
    centroids = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])

            # Track people only (matches sample output)
            if class_id == PERSON_CLASS_ID and confidence > CONF_THRESHOLD:
                box = detection[0:4] * np.array([W, H, W, H])
                (center_x, center_y, width, height) = box.astype("int")
                x = int(center_x - (width / 2))
                y = int(center_y - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(confidence)
                centroids.append((int(center_x), int(center_y)))

    # Apply Non-Maximum Suppression (NMS) to filter out overlapping boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)

    nms_centroids = []
    nms_boxes = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            nms_centroids.append(centroids[i])
            nms_boxes.append(boxes[i])

    # Update the tracker with the centroids of detected objects
    objects = tracker.update(nms_centroids)
    paths = tracker.get_paths()

    # Map current centroids back to their NMS boxes for drawing
    centroid_to_box = {c: b for c, b in zip(nms_centroids, nms_boxes)}

    # Draw bounding boxes, trails, and labels on the frame
    draw_paths(frame, paths, max_len=30)

    for objectID, centroid in objects.items():
        color = [int(c) for c in colors[int(objectID) % len(colors)]]
        cx, cy = int(centroid[0]), int(centroid[1])

        box = centroid_to_box.get(tuple(centroid))
        if box is not None:
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        else:
            # Object is being coasted (lost briefly) — still mark centroid
            cv2.circle(frame, (cx, cy), 4, color, -1)

        cv2.putText(
            frame,
            f"ID {objectID}",
            (cx - 10, cy - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
        cv2.circle(frame, (cx, cy), 3, color, -1)

    # Write the processed frame to the output video
    out_vid.write(frame)

    # Optional live preview (comment out if running headless)
    # cv2.imshow("YOLO Path Tracker", frame)
    # if cv2.waitKey(1) & 0xFF == ord("q"):
    #     break

# Release video writer and video capture objects
out_vid.release()
cap.release()
cv2.destroyAllWindows()
print("Saved annotated video to output.mp4")
