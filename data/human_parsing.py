import numpy as np
from torch.utils.data.dataset import Dataset
from glob import glob
import os
import cv2


def preprocess_dataset(root):
    import scipy.io as scio
    data = scio.loadmat(r"D:\temp_data\instance-level_human_parsing\mpii_human_pose_v1_u12_1.mat")
    indices = []
    txt_path = os.path.join(root, "train_id.txt")
    with open(txt_path) as f:
        for line in f:
            indices.append(line.strip())
    for name in indices:
        image = cv2.imread(os.path.join(root, f"Images/{name}.jpg"))
        body = cv2.imread(os.path.join(root, f"Human_ids/{name}.png"))[..., 0]
        face = cv2.imread(os.path.join(root, f"Category_ids/{name}.png"))[..., 0]

        body_indices = np.unique(body)
        for idx in body_indices:
            if idx == 0:
                continue
            mask = (body == idx).astype(np.uint8)
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            lines = np.concatenate(contours, 0)
            # lines = sorted(contours, key=lambda x: cv2.contourArea(x))[-1]
            minx = lines[..., 0].min()
            miny = lines[..., 1].min()
            maxx = lines[..., 0].max()
            maxy = lines[..., 1].max()
            cv2.rectangle(image, (minx, miny), (maxx, maxy), (0, 255, 0), 2)
        cv2.imshow("image", image)
        cv2.waitKey()


class HumanParsing(Dataset):
    def __init__(self, root, ):
        self.root = root


if __name__ == '__main__':
    preprocess_dataset(r"D:\temp_data\instance-level_human_parsing\Training")