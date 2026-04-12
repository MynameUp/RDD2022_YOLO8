from ultralytics import YOLO
import cv2

model = YOLO("./runs/detect/runs/rdd2022/yolov8_road_damage/weights/best.pt")
results = model("../RDD2022/China_MotorBike/test/images/China_MotorBike_001978.jpg", conf=0.25)

for r in results:
    img = r.plot()
    cv2.imshow("../results", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()