import cv2
import numpy as np
from ultralytics import YOLO
from scipy.spatial import distance as dist
from collections import OrderedDict
from flask import Flask, Response, render_template, jsonify

# Initialize Flask
app = Flask(__name__)

# Initialize the YOLOv8 model
model = YOLO('survei2.pt')  # Make sure the model is located in the correct path

# Use the default (built-in) camera. Change the index if needed.
# For example, use 1 for an external camera, etc.
cap = cv2.VideoCapture(0)

# Centroid Tracker class
class CentroidTracker:
    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        # If no bounding boxes (rects) are provided, mark all existing objects as disappeared.
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        # Compute the centroids of the bounding boxes
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        # If no objects are currently being tracked, register all centroids
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            # Grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # Compute the distance between each pair of object centroids and input centroids
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            # Find the smallest value in each row, then sort row indices based on the minimum values
            rows = D.min(axis=1).argsort()
            # Find the column index that has the smallest value for each row
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            # Loop over the combination of (row, col) index tuples
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                objectID = objectIDs[row]
                # Update the centroid for the object we have matched
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)

            # Compute both the row and column indices we have not yet used
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # If the number of object centroids is greater than or equal to the number of input centroids
            # check and see if some of these objects have potentially disappeared
            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                # Register each new centroid as a trackable object
                for col in unusedCols:
                    self.register(inputCentroids[col])

        return self.objects

# Initialize the Centroid Tracker
dt = CentroidTracker()
total_count = 0
current_count = 0

# Function to generate video frames
def generate_frame():
    global total_count, current_count
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects with YOLO
        results = model(frame)
        persons = [d for d in results[0].boxes.data if int(d[5]) == 0]
        rects = []

        # Draw bounding boxes for each detected object
        for person in persons:
            x1, y1, x2, y2 = map(int, person[:4])
            rects.append((x1, y1, x2, y2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Update the tracker
        objects = dt.update(rects)
        current_count = len(objects)

        if dt.nextObjectID > total_count:
            total_count = dt.nextObjectID

        for objectID, centroid in objects.items():
            text = f"ID {objectID}"
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        # Convert the frame to JPEG format
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Send the frame to the browser
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route for the main page
@app.route('/')
def index():
    return render_template('hasilcount.html')

# Route for the video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route for the number of people data
@app.route('/count_data')
def count_data():
    data = {
        "current_count": current_count,
        "total_count": total_count
    }
    return jsonify(data)

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
