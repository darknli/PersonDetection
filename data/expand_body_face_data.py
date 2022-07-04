from glob import glob
import os
import json
from augmentations import RandomCrop
import cv2
import numpy as np


class BodyFaceData:
    def __init__(self, begin_idx, root):
        """
        人体/人脸数据扩充包
        Parameters
        ----------
        begin_idx: 起始索引号
        """
        self.begin_idx = begin_idx
        self.classes = ["01", "02"]
        self.anno = self._get_anno(root)
        self.rc = RandomCrop(640)

    def _get_anno(self, root):
        files = glob(os.path.join(root, "*.json"))

        anno = []
        for file in files:
            with open(file, encoding="utf-8") as f:
                text = json.loads(f.read())
            boxes = []
            for item in text["shapes"]:
                # label=4的话变成3，因为压根没有3
                b = [self.classes.index(item["label"])] + item["points"][0] + item["points"][1]
                boxes.append(b)

            image_path = file.replace("json", "jpg")
            if not os.path.exists(image_path):
                continue
            anno.append(
                {
                    "filename": image_path,
                    "boxes": np.array(boxes)
                }
            )
        return anno

    def __len__(self):
        return len(self.anno) * 10

    def get(self, idx):
        assert self.begin_idx <= idx
        item = self.anno[(idx - self.begin_idx)//10]
        image = cv2.imread(item["filename"])
        boxes = item["boxes"]

        # draw_image = image.copy()
        # for box in boxes:
        #     x1, y1, x2, y2 = box[1:].astype(int)
        #     if box[0] == 0:
        #         c = (0, 255, 0)
        #     else:
        #         c = (255, 0, 0)
        #     cv2.rectangle(draw_image, (x1, y1), (x2, y2), c, 1, cv2.LINE_4)
        # cv2.imshow("before", draw_image)

        image, boxes = self.rc((image, boxes))
        # for box in boxes:
        #     x1, y1, x2, y2 = box[1:].astype(int)
        #     if box[0] == 0:
        #         c = (0, 255, 0)
        #     else:
        #         c = (255, 0, 0)
        #     cv2.rectangle(image, (x1, y1), (x2, y2), c, 1, cv2.LINE_4)
        # cv2.imshow("image", image)
        # cv2.waitKey()
        return image, boxes


if __name__ == '__main__':
    expand_data = BodyFaceData(0, "D:/temp_data/labelme_bodyface")

    for i in range(len(expand_data)):
        expand_data.get(i)