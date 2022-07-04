import os
from glob import glob
import cv2
import json
import numpy as np


def check_anno_correct(root, dst):
    if os.path.isdir(root):
        files = glob(os.path.join(root, "*.jpg"))
    else:
        files = [root]

    for file in files:
        print(file)
        image = cv2.imread(file)
        json_path = file.replace("jpg", "json")
        if os.path.exists(json_path):
            with open(json_path, encoding="utf-8") as f:
                ori_code = json.loads(f.read())
            points = - np.ones((4, 2))
            for point in ori_code["shapes"]:
                x, y = point["points"][0]
                cls = int(point["label"]) - 1
                points[cls] = (x, y)

            width = int(points[:, 0].max() - points[:, 0].min())
            height = int(points[:, 1].max() - points[:, 1].min())
            after = [
                (0, 0),
                (width, 0),
                (0, height),
                (width, height)
            ]
            after = np.array(after, dtype=np.float32)
            mat = cv2.getPerspectiveTransform(points.astype(np.float32), after)
            image = cv2.warpPerspective(image, mat, (width, height))
        cv2.imwrite(os.path.join(dst, "cor_"+os.path.basename(file)), image)
        print(f"finish {os.path.join(dst, 'cor_'+os.path.basename(file))}")


if __name__ == '__main__':
    # dirs = glob("E:\Data\OCR\my_code\photo\correct\*")
    dirs = [r"E:\Data\OCR\my_code\target\PackingGeneralML"]
    for path in dirs:
        check_anno_correct(path, r"E:\Data\OCR\my_code\screenshot\backup")
