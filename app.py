import cv2
import numpy as np
from ultralytics import YOLO
from scipy.spatial import distance as dist
from collections import OrderedDict
from flask import Flask, Response, render_template, jsonify

# Inisialisasi Flask
app = Flask(__name__)
# Inisialisasi YOLOv8 model

model = YOLO('survei2.pt') 
# Pastikan model ini ada di lokasi yang benar

#video_path = 'D:/TEL U/STAS-RG/Computer Vision/coba detection/Video/DJI_0027.MP4'

cap = cv2.VideoCapture(1)
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
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        return self.objects


# Inisialisasi Centroid Tracker
dt = CentroidTracker()
total_count = 0
current_count = 0


# Fungsi untuk mengenerate frame video
def generate_frame():
    global total_count, current_count
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Deteksi objek dengan YOLO
        results = model(frame)
        persons = [d for d in results[0].boxes.data if int(d[5]) == 0]
        rects = []

        # Gambar bounding box untuk setiap objek yang terdeteksi
        for person in persons:
            x1, y1, x2, y2 = map(int, person[:4])
            rects.append((x1, y1, x2, y2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Update tracker
        objects = dt.update(rects)
        current_count = len(objects)

        if dt.nextObjectID > total_count:
            total_count = dt.nextObjectID

        for objectID, centroid in objects.items():
            text = f"ID {objectID}"
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
    

        # Konversi frame ke format JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Kirim frame ke browser
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# Route untuk halaman utama
@app.route('/')
def index():
    return render_template('hasilcount.html')


# Route untuk video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Route untuk data jumlah orang
@app.route('/count_data')
def count_data():
    data = {
        "current_count": current_count,
        "total_count": total_count
    }
    return jsonify(data)


# Jalankan aplikasi Flask
if __name__ == "__main__":
    app.run(debug=True)
