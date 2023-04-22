import os
import json
import cv2

folder_path = os.getcwd()  # 当前文件夹路径

for file_name in os.listdir(folder_path):
    # 检查文件名是否以".jpg"结尾
    if file_name.endswith(".jpg"):
        json_file = file_name[:-4] + ".json"  # 对应的JSON文件名
        # 如果JSON文件存在，则进行处理
        if os.path.exists(os.path.join(folder_path, json_file)):
            # 从JSON文件中读取检测结果
            with open(os.path.join(folder_path, json_file), "r") as f:
                detections = json.load(f)

            # 从JPG文件中读取图像
            img = cv2.imread(os.path.join(folder_path, file_name))

            # 绘制边界框和标签
            for detection in detections:
                if detection["label"] != "background":
                    x1, y1, x2, y2 = map(int, detection["box"])
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        img,
                        detection["label"],
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2,
                    )

            # 保存可视化结果
            mask_file = os.path.join(folder_path, file_name[:-4] + "_mask.jpg")
            cv2.imwrite(mask_file, img)
