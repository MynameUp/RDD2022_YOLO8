from ultralytics import YOLO

def main():
    model = YOLO("./runs/detect/runs/rdd2022/yolov8_road_damage/weights/best.pt")

    """
    # 返回的是检测结果列表
    results[0].boxes         # 检测框坐标和类别
    results[0].plot()        # 绘制好的图片数组
    # 保存的检测图片（带可视化框）
    """
    results = model.predict(
        source="../datasets/rdd2022_yolo/images/val",
        imgsz=640,
        conf=0.25,
        save=True,
        project="runs/rdd2022",
        name="predict_results",
        device=0
    )

    print("预测完成！结果已保存。")

if __name__ == "__main__":
    main()