import os
import random
import shutil
import xml.etree.ElementTree as ET
from glob import glob
from tqdm import tqdm

# 类别映射，可根据你的数据集实际类别调整
CLASS_MAP = {
    "D00": 0,
    "D10": 1,
    "D20": 2,
    "D40": 3
}
# , "India", "Japan", "Czech"
COUNTRIES = ["China_MotorBike"]

def convert_bbox(size, box):
    """
    将 VOC 格式边界框转换为 YOLO 格式
    size: (width, height)
    box: (xmin, ymin, xmax, ymax)
    return: (x_center, y_center, w, h)
    """
    w_img, h_img = size
    xmin, ymin, xmax, ymax = box

    x_center = ((xmin + xmax) / 2.0) / w_img
    y_center = ((ymin + ymax) / 2.0) / h_img
    w = (xmax - xmin) / w_img
    h = (ymax - ymin) / h_img

    return x_center, y_center, w, h

def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    objects = []
    for obj in root.findall("object"):
        cls_name = obj.find("name").text.strip()
        if cls_name not in CLASS_MAP:
            continue

        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        bbox = convert_bbox((width, height), (xmin, ymin, xmax, ymax))
        objects.append((CLASS_MAP[cls_name], bbox))

    return objects

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def split_train_val(xml_files, val_ratio=0.2, seed=42):
    """
    将文件列表分割为训练集和验证集

    Args:
        xml_files: XML文件列表
        val_ratio: 验证集比例，默认0.2（即80%训练，20%验证）
        seed: 随机种子，确保可重复性

    Returns:
        train_files: 训练集文件列表
        val_files: 验证集文件列表
    """
    random.seed(seed)
    shuffled_files = xml_files.copy()
    random.shuffle(shuffled_files)

    split_idx = int(len(shuffled_files) * (1 - val_ratio))
    train_files = shuffled_files[:split_idx]
    val_files = shuffled_files[split_idx:]

    print(f"数据集分割完成: 训练集 {len(train_files)} 张, 验证集 {len(val_files)} 张")
    return train_files, val_files

def process_split(src_root, dst_root, split="train", xml_files=None):
    """
    处理数据集分割

    Args:
        src_root: 源数据根目录
        dst_root: 目标数据根目录
        split: 分割类型（"train" 或 "val"）
        xml_files: 指定要处理的XML文件列表，如果为None则处理所有文件
    """
    img_out_dir = os.path.join(dst_root, "images", split)
    label_out_dir = os.path.join(dst_root, "labels", split)

    ensure_dir(img_out_dir)
    ensure_dir(label_out_dir)

    for country in COUNTRIES:
        img_dir = os.path.join(src_root, country, "train", "images")
        xml_dir = os.path.join(src_root, country, "train", "annotations", "xmls")

        if not os.path.exists(img_dir) or not os.path.exists(xml_dir):
            print(f"[Warning] 跳过不存在路径: {country} train")
            continue

        # 如果没有指定文件列表，则获取所有XML文件
        if xml_files is None:
            xml_files_all = glob(os.path.join(xml_dir, "*.xml"))
            xml_files_to_process = xml_files_all
        else:
            # 过滤出当前国家的文件
            xml_files_to_process = [f for f in xml_files if country in f]

        if not xml_files_to_process:
            print(f"[Warning] {country} 没有找到需要处理的文件")
            continue

        for xml_file in tqdm(xml_files_to_process, desc=f"{country}-{split}"):
            base_name = os.path.splitext(os.path.basename(xml_file))[0]

            # 找对应图片，可能是 jpg/png
            image_file = None
            for ext in [".jpg", ".jpeg", ".png"]:
                candidate = os.path.join(img_dir, base_name + ext)
                if os.path.exists(candidate):
                    image_file = candidate
                    break

            if image_file is None:
                print(f"[Warning] 未找到图片文件: {base_name}")
                continue

            objects = parse_xml(xml_file)

            # 为避免不同国家重名，加国家前缀
            new_base_name = f"{country}_{base_name}"
            image_ext = os.path.splitext(image_file)[1]

            dst_image_path = os.path.join(img_out_dir, new_base_name + image_ext)
            dst_label_path = os.path.join(label_out_dir, new_base_name + ".txt")

            shutil.copy2(image_file, dst_image_path)

            with open(dst_label_path, "w", encoding="utf-8") as f:
                for cls_id, (x, y, w, h) in objects:
                    f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

def main():
    src_root = "../RDD2022"              # 原始数据集路径
    dst_root = "../datasets/rdd2022_yolo"  # 转换后的 YOLO 数据路径

    # 设置验证集比例
    val_ratio = 0.2  # 20%作为验证集，80%作为训练集

    print("开始处理数据集...")

    for country in COUNTRIES:
        xml_dir = os.path.join(src_root, country, "train", "annotations", "xmls")

        if not os.path.exists(xml_dir):
            print(f"[Warning] 跳过不存在的国家: {country}")
            continue

        # 获取所有XML文件
        all_xml_files = glob(os.path.join(xml_dir, "*.xml"))
        print(f"\n{country}: 共找到 {len(all_xml_files)} 个标注文件")

        # 分割训练集和验证集
        train_files, val_files = split_train_val(all_xml_files, val_ratio=val_ratio)

        # 处理训练集
        print("\n处理训练集...")
        process_split(src_root, dst_root, split="train", xml_files=train_files)

        # 处理验证集
        print("\n处理验证集...")
        process_split(src_root, dst_root, split="val", xml_files=val_files)

    print("\n" + "="*50)
    print("RDD2022 转换为 YOLO 格式完成！")
    print(f"验证集比例: {val_ratio*100}%")
    print(f"输出目录: {os.path.abspath(dst_root)}")
    print("="*50)

if __name__ == "__main__":
    main()