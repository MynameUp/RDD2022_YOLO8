import yaml
import os

def main():
    dataset_root = os.path.abspath("../datasets/rdd2022_yolo")

    data = {
        "path": dataset_root,
        "train": "images/train",
        "val": "images/val",
        "names": {
            0: "D00",
            1: "D10",
            2: "D20",
            3: "D40"
        }
    }

    yaml_path = os.path.join(dataset_root, "dataset.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)

    print(f"dataset.yaml 已生成: {yaml_path}")

if __name__ == "__main__":
    main()