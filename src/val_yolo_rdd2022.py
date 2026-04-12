# from ultralytics import YOLO
#
# def main():
#     model = YOLO("runs/rdd2022/yolov8_road_damage/weights/best.pt")
#
#     metrics = model.val(
#         data="datasets/rdd2022_yolo/dataset.yaml",
#         imgsz=640,
#         batch=16,
#         device=0
#     )
#
#     print(metrics)
#
# if __name__ == "__main__":
#     main()
from ultralytics import YOLO

def main():
    model = YOLO("./runs/detect/runs/rdd2022/yolov8_road_damage/weights/best.pt")

    """
    # 返回的是统计指标对象
    metrics.box.map          # mAP50-95 (平均精度均值)
    metrics.box.map50        # mAP@IoU=0.50
    metrics.box.map75        # mAP@IoU=0.75
    metrics.box.maps         # 各类别的mAP
    metrics.box.precision    # 精确率
    metrics.box.recall       # 召回率
    metrics.speed            # 推理速度统计
    """
    metrics = model.val(
        data="../datasets/rdd2022_yolo/dataset.yaml",
        imgsz=640,
        batch=16,
        device=0,
        split="val",
        plots=False
    )

    print("验证完成！")
    print(metrics)

if __name__ == "__main__":
    main()