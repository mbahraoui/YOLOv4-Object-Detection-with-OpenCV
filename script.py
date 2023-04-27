import cv2
import numpy as np

net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")

classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

colors = np.random.uniform(0, 255, size=(len(classes), 3))

img = cv2.imread("image.jpeg")

# Resize image
height, width = 1440, 1440  # Set the desired height and width
img = cv2.resize(img, (width, height))

blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), swapRB=True, crop=False)

net.setInput(blob)

outputs = net.forward(net.getUnconnectedOutLayersNames())

boxes = []
confidences = []
class_ids = []
for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * img.shape[1])
            center_y = int(detection[1] * img.shape[0])
            width = int(detection[2] * img.shape[1])
            height = int(detection[3] * img.shape[0])
            left = int(center_x - width / 2)
            top = int(center_y - height / 2)
            boxes.append([left, top, width, height])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

for i in indices:
    box = boxes[i]
    left = box[0]
    top = box[1]
    width = box[2]
    height = box[3]
    color = colors[class_ids[i]]
    cv2.rectangle(img, (left, top), (left + width, top + height), color, 2)
    cv2.putText(img, classes[class_ids[i]], (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
