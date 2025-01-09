import cv2 as cv
import numpy as np
from ultralytics import YOLO
import os

car_detections = []
bboxes = []
iou_car = []
txt_path = os.path.join('KITTI_Selection/labels/006227.txt')
intrensic_matric_path = os.path.join('KITTI_Selection/calib/006227.txt')

def cal_distance(camera_height=1.65):
    for index in range(len(bboxes)):
        x1,y1,x2,y2 = bboxes[index]
        x_mid = (x1+x2)/2
        y_mid = y2

        # Calculate Depth
        matrix = np.loadtxt(intrensic_matric_path)
        focal_length = matrix[1][1]
        vertical_point = matrix[1][2]
        depth = (focal_length * camera_height)/(y_mid - vertical_point)
        print('Depth',depth)

def txt_boundingBox():
    with open(txt_path, 'r') as f:
        for line in f:
            data = line.strip().split()
            x1, y1, x2, y2 = map(float, data[1:-1])
            bboxes.append([x1, y1, x2, y2])

def calculate_IOU():
    global car_detections,txt_path,bboxes,iou_car
    for index in range(len(car_detections)):
        detected_box = car_detections[index]
        ground_detected_box = bboxes[index]
        print(detected_box,"-------",ground_detected_box)

        x_int_min = max(detected_box[0], ground_detected_box[0])
        y_int_min = max(detected_box[1], ground_detected_box[1])
        x_int_max = min(detected_box[2], ground_detected_box[2])
        y_int_max = min(detected_box[3], ground_detected_box[3])

        width = max(0,x_int_max-x_int_min)
        height = max(0,y_int_max-y_int_min)

        Area_int = width * height

        Area1 = (detected_box[2]-detected_box[0]) * (detected_box[3] - detected_box[1])
        Area2 = (ground_detected_box[2] - ground_detected_box[0]) * (ground_detected_box[3] - ground_detected_box[1])

        Area_uni = Area1+Area2-Area_int

        iou = Area_int/Area_uni
        iou_car.append(iou)
    print(iou_car)

def precision_and_recall(car_detections,bboxes,iou_threshold=0.5):
    tp,fp,fn = 0,0,0
    right_predbox = set()

    for car_detection in car_detections:
        matched = False
        for index,bbox in enumerate(bboxes):
            if index in right_predbox:
                continue
            iou = iou_car[index]
            if iou >= iou_threshold:
                tp+=1
                right_predbox.add(index)
                matched = True
                break
        if not matched:
            fp+=1

    fn =  len(bboxes) - len(right_predbox)

    precision = tp/(tp+fp) if tp + fp > 0 else 0
    recall = tp/(tp+fn) if tp + fn > 0 else 0

    return precision,recall

def detect_car(detections,image):
    global car_detections
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        print('Detections', detection)
        if class_id == 2:
            car_detections.append([x1, y1, x2, y2])
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

def train_model(image):
    model = YOLO('yolov8n.pt')
    detections = model(image)[0]
    return detections

image = cv.imread('KITTI_Selection/images/006227.png')
detections = train_model(image)
detect_car(detections,image)
txt_boundingBox()
calculate_IOU()
precision,recall = precision_and_recall(car_detections,bboxes,iou_threshold=0.5)
print('Precision and Recall:',precision,'----',recall)
cal_distance(camera_height=1.65)
# Display results
cv.imshow('window',image)
cv.waitKey(0)
