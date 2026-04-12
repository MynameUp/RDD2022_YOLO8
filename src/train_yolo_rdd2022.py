from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data="../datasets/rdd2022_yolo/dataset.yaml",
        epochs=50,                 #训练轮数
        imgsz=640,                  #训练图片尺寸
        batch=16,                   #训练批次
        workers=0,                  #数据加载线程数，Windows系统建议设为0
        device=0,                   # 如果没有GPU可改成 'cpu'
        project="runs/rdd2022",     # 训练结果保存路径
        name="yolov8_road_damage",  # 模型保存名称
        pretrained=True,            # 是否使用预训练模型

        # 显式指定优化器参数
        optimizer="SGD",           # 可选: SGD, Adam, AdamW, auto
        lr0=0.01,                  # 初始学习率
        lrf=0.01,                  # 最终学习率比例
        momentum=0.937,            #动量参数
        weight_decay=0.0005,       #权重衰减

        # 训练控制
        patience=20,               # 早停耐心值，若验证指标在patience轮内没有提升则停止训练
        save=True,                 # 保存模型
        save_period=10,            # 每10个epoch额外保存一次
        val=True,                  # 每轮训练后进行验证
        verbose=True,              # 是否打印训练过程
    )

    print("训练完成，best.pt 会自动保存在 weights 目录下。")

if __name__ == "__main__":
    main()