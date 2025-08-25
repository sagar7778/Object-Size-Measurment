import cv2
import numpy as np
import imutils
from imutils import perspective
from flask import Flask, render_template, Response, jsonify
import datetime
import json

app = Flask(__name__)

# ---------- Storage ----------
scanned_results = []
product_counter = 1
new_entry_flag = False

# ---------- Flags ----------
qr_detector = cv2.QRCodeDetector()    # OpenCV QR detector
qr_detected = set()                   # Store already scanned QR data

# ---------- Utility ----------
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def save_to_json():
    with open('scanned_results.json', 'w') as json_file:
        serializable_results = []
        for result in scanned_results:
            serializable_result = {
                "type": result["type"],
                "timestamp": result["timestamp"]
            }
            if result["type"] == "object":
                serializable_result.update({
                    "name": result["name"],
                    "height_cm": float(result["height_cm"]),
                    "width_cm": float(result["width_cm"])
                })
            elif result["type"] == "qr":
                serializable_result.update({
                    "data": result["data"]
                })
            serializable_results.append(serializable_result)
        json.dump(serializable_results, json_file, indent=4)

# ---------- Object Measurement ----------
def measure_frame(frame, ref_width_mm=85.6):
    global scanned_results, product_counter, new_entry_flag
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    pixelsPerMetric = None
    for c in cnts:
        if cv2.contourArea(c) < 1000:
            continue

        box = cv2.minAreaRect(c)
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)

        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        dA = np.linalg.norm(np.array([tltrX, tltrY]) - np.array([blbrX, blbrY]))
        dB = np.linalg.norm(np.array([tlblX, tlblY]) - np.array([trbrX, trbrY]))

        if pixelsPerMetric is None:
            pixelsPerMetric = dB / ref_width_mm  # Reference card width in mm

        height_cm = (dA / pixelsPerMetric) / 10
        width_cm = (dB / pixelsPerMetric) / 10

        cv2.drawContours(frame, [box.astype("int")], -1, (0, 255, 0), 2)
        cv2.putText(frame, f"{height_cm:.2f}cm x {width_cm:.2f}cm",
                    (int(tltrX - 40), int(tltrY - 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        product_name = f"Product-{product_counter}"
        result = {
            "type": "object",
            "name": product_name,
            "height_cm": height_cm,
            "width_cm": width_cm,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        scanned_results.append(result)
        new_entry_flag = True
        print(f"Stored Object: {result}")
        product_counter += 1
        break  # Remove this line if you want to detect multiple objects in a single frame

    return frame

# ---------- QR Detection ----------
def detect_qr(frame):
    global scanned_results, new_entry_flag, qr_detected
    data, points, _ = qr_detector.detectAndDecode(frame)
    if points is not None and data:
        if data in qr_detected:  # Already scanned
            return frame

        pts = points[0].astype(int)
        for j in range(len(pts)):
            cv2.line(frame, tuple(pts[j]), tuple(pts[(j + 1) % len(pts)]), (0, 0, 255), 2)

        cv2.putText(frame, f"QR: {data}",
                    (pts[0][0], pts[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        result = {
            "type": "qr",
            "data": data,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        scanned_results.append(result)
        new_entry_flag = True
        qr_detected.add(data)  # Mark as scanned
        print(f"Stored QR: {result}")

    return frame

# ---------- Video Generator ----------
def generate_frames():
    global new_entry_flag
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use CAP_DSHOW for Windows
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = detect_qr(frame)
        frame = measure_frame(frame)

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    cap.release()

# ---------- Routes ----------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video")
def video():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/data")
def get_data():
    global new_entry_flag
    if new_entry_flag:
        new_entry_flag = False
        save_to_json()
        return jsonify(scanned_results)
    return jsonify([])

# ---------- Main ----------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
