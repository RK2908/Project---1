import torch
import cv2
import time
import RPi.GPIO as GPIO
import numpy as np
from datetime import datetime
import os
import serial

GPIO.setwarnings(False)


# ------------------- GPIO SETUP -------------------
TRIG = 23
ECHO = 24
BUZZER = 18

GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)
GPIO.setup(BUZZER, GPIO.OUT)
GPIO.output(BUZZER, GPIO.LOW)

# ------------------- GSM SETUP -------------------
gsm = serial.Serial("/dev/serial0", baudrate=9600, timeout=1)
def send_sms(message, phone_number="+91XXXXXXXXXX"):
    print("Sending SMS...")
    gsm.write(b'AT\r')
    time.sleep(1)
    gsm.write(b'AT+CMGF=1\r')  # Set SMS text mode
    time.sleep(1)
    gsm.write(f'AT+CMGS="{phone_number}"\r'.encode())
    time.sleep(1)
    gsm.write((message + "\x1A").encode())  # \x1A is ASCII code for Ctrl+Z (end of message)
    time.sleep(3)
    print("SMS sent.")

# ------------------- YOLOv5 MODEL LOAD -------------------
print("Loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()
person_class_id = 0
print("Model loaded successfully.")

# ------------------- CAMERA SETUP -------------------
cap = cv2.VideoCapture(0)
time.sleep(2)

# ------------------- IMAGE SAVE PATH -------------------
SAVE_DIR = "captures"
os.makedirs(SAVE_DIR, exist_ok=True)

# ------------------- DISTANCE FUNCTION -------------------
def get_distance():
    GPIO.output(TRIG, False)
    time.sleep(0.05)

    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    pulse_start = time.time()
    timeout = pulse_start + 0.04
    while GPIO.input(ECHO) == 0 and time.time() < timeout:
        pulse_start = time.time()

    pulse_end = time.time()
    timeout = pulse_end + 0.04
    while GPIO.input(ECHO) == 1 and time.time() < timeout:
        pulse_end = time.time()

    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17150
    return round(distance, 2)

# ------------------- MAIN LOOP -------------------
last_sms_time = 0
sms_interval = 60  # seconds

try:
    while True:
        distance = get_distance()
        print(f"Distance: {distance} cm")

        ret, frame = cap.read()
        if not ret or frame is None:
            print("Failed to capture frame.")
            continue

        cv2.imshow("Live Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if distance <= 30:
            print("Object detected within 30cm")

            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except cv2.error as e:
                print("cv2 error:", e)
                continue

            results = model(frame_rgb)
            person_detected = False

            for *box, conf, cls in results.xyxy[0]:
                if int(cls) == person_class_id:
                    person_detected = True
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f'Person {conf:.2f}'
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if person_detected:
                GPIO.output(BUZZER, GPIO.HIGH)
                print("Person detected - BUZZER ON")

                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = os.path.join(SAVE_DIR, f"person_{timestamp}.jpg")
                cv2.imwrite(filename, frame)
                print(f"Image saved: {filename}")

                # Send SMS only once every `sms_interval` seconds
                if time.time() - last_sms_time > sms_interval:
                    sms_message = f"Alert! Children detected near borewell at {timestamp}"
                    send_sms(sms_message, phone_number="+916374587367")
                    send_sms(sms_message, phone_number="+918870778134")  
                    last_sms_time = time.time()
                GPIO.output(BUZZER, GPIO.LOW)
            else:
                GPIO.output(BUZZER, GPIO.LOW)
                print("No person - BUZZER OFF")

except KeyboardInterrupt:
    print("Stopped by User")

finally:
    cap.release()
    cv2.destroyAllWindows()
    GPIO.cleanup()
