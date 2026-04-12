import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    result_dir = "./runs/detect/runs/rdd2022/yolov8_road_damage"
    csv_file = os.path.join(result_dir, "results.csv")

    if not os.path.exists(csv_file):
        print(f"未找到训练日志文件: {csv_file}")
        return

    df = pd.read_csv(csv_file)
    df.columns = [c.strip() for c in df.columns]  # 去掉列名空格

    print("可用列名：")
    for col in df.columns:
        print(col)

    # 创建保存目录
    save_dir = os.path.join(result_dir, "custom_plots")
    os.makedirs(save_dir, exist_ok=True)


# ... existing code ...
    # ========== 1. Precision / Recall / mAP ==========
    plt.figure(figsize=(10, 6))

    if "metrics/precision(B)" in df.columns:
        plt.plot(df["epoch"].values, df["metrics/precision(B)"].values, label="Precision", linewidth=2)
    if "metrics/recall(B)" in df.columns:
        plt.plot(df["epoch"].values, df["metrics/recall(B)"].values, label="Recall", linewidth=2)
    if "metrics/mAP50(B)" in df.columns:
        plt.plot(df["epoch"].values, df["metrics/mAP50(B)"].values, label="mAP@0.5", linewidth=2)
    if "metrics/mAP50-95(B)" in df.columns:
        plt.plot(df["epoch"].values, df["metrics/mAP50-95(B)"].values, label="mAP@0.5:0.95", linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.title("Training Metrics Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "metrics_curve.png"), dpi=300)
    plt.close()

    # ========== 2. Loss 曲线 ==========
    plt.figure(figsize=(10, 6))

    for col in ["train/box_loss", "train/cls_loss", "train/dfl_loss",
                "val/box_loss", "val/cls_loss", "val/dfl_loss"]:
        if col in df.columns:
            plt.plot(df["epoch"].values, df[col].values, label=col, linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=300)
    plt.close()

    # ========== 3. 单独画 Precision ==========
    if "metrics/precision(B)" in df.columns:
        plt.figure(figsize=(8, 5))
        plt.plot(df["epoch"].values, df["metrics/precision(B)"].values, color="blue", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Precision")
        plt.title("Precision Curve")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "precision_curve.png"), dpi=300)
        plt.close()

    # ========== 4. 单独画 Recall ==========
    if "metrics/recall(B)" in df.columns:
        plt.figure(figsize=(8, 5))
        plt.plot(df["epoch"].values, df["metrics/recall(B)"].values, color="green", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Recall")
        plt.title("Recall Curve")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "recall_curve.png"), dpi=300)
        plt.close()

    # ========== 5. 单独画 mAP ==========
    plt.figure(figsize=(8, 5))
    has_map = False

    if "metrics/mAP50(B)" in df.columns:
        plt.plot(df["epoch"].values, df["metrics/mAP50(B)"].values, label="mAP@0.5", linewidth=2)
        has_map = True
    if "metrics/mAP50-95(B)" in df.columns:
        plt.plot(df["epoch"].values, df["metrics/mAP50-95(B)"].values, label="mAP@0.5:0.95", linewidth=2)
        has_map = True

    if has_map:
        plt.xlabel("Epoch")
        plt.ylabel("mAP")
        plt.title("mAP Curve")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "map_curve.png"), dpi=300)
        plt.close()
# ... existing code ...


    print(f"可视化完成，结果保存在: {save_dir}")

if __name__ == "__main__":
    main()