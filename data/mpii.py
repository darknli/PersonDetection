import numpy as np
from torch.utils.data.dataset import Dataset
from glob import glob
import os
import cv2
from scipy.io import loadmat


def preprocess_dataset(root):
    import scipy.io as scio
    data = scio.loadmat(r"D:\temp_data\MPII\mpii_human_pose_v1_u12_2\mpii_human_pose_v1_u12_1.mat")
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


def save_joints():
    mat = loadmat(r'D:\temp_data\MPII\mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat')
    fout = open(r'D:\temp_data\MPII\mpii_human_pose_v1_u12_2\mpii_list.txt', 'w')
    filename_set = {}
    for i, (anno, train_flag) in enumerate(
            zip(mat['RELEASE']['annolist'][0, 0][0],
                mat['RELEASE']['img_train'][0, 0][0],
                )):
        img_fn = anno['image']['name'][0, 0][0]
        train_flag = int(train_flag)

        head_rect = []
        if 'x1' in str(anno['annorect'].dtype):
            head_rect = zip(
                [x1[0, 0] for x1 in anno['annorect']['x1'][0]],
                [y1[0, 0] for y1 in anno['annorect']['y1'][0]],
                [x2[0, 0] for x2 in anno['annorect']['x2'][0]],
                [y2[0, 0] for y2 in anno['annorect']['y2'][0]])

        if 'annopoints' in str(anno['annorect'].dtype):
            # only one person
            annopoints = anno['annorect']['annopoints'][0]
            head_x1s = anno['annorect']['x1'][0]
            head_y1s = anno['annorect']['y1'][0]
            head_x2s = anno['annorect']['x2'][0]
            head_y2s = anno['annorect']['y2'][0]

            for annopoint, head_x1, head_y1, head_x2, head_y2 in zip(
                    annopoints, head_x1s, head_y1s, head_x2s, head_y2s):
                if annopoint != []:
                    head_rect = [float(head_x1[0, 0]),
                                 float(head_y1[0, 0]),
                                 float(head_x2[0, 0]),
                                 float(head_y2[0, 0])]
                    # build feed_dict
                    feed_dict = {}
                    feed_dict['width'] = int(abs(float(head_x2[0, 0]) - float(head_x1[0, 0])))
                    feed_dict['height'] = int(abs(float(head_y2[0, 0]) - float(head_y1[0, 0])))

                    # joint coordinates
                    annopoint = annopoint['point'][0, 0]
                    j_id = [str(j_i[0, 0]) for j_i in annopoint['id'][0]]
                    x = [x[0, 0] for x in annopoint['x'][0]]
                    y = [y[0, 0] for y in annopoint['y'][0]]
                    joint_pos = {}
                    for _j_id, (_x, _y) in zip(j_id, zip(x, y)):
                        joint_pos[str(_j_id)] = [float(_x), float(_y)]

                    # visiblity list
                    if 'is_visible' in str(annopoint.dtype):
                        vis = [v[0] if v else [0]
                               for v in annopoint['is_visible'][0]]
                        vis = dict([(k, int(v[0])) if len(v) > 0 else v
                                    for k, v in zip(j_id, vis)])
                    else:
                        vis = None
                    feed_dict['x'] = x
                    feed_dict['y'] = y
                    feed_dict['vis'] = vis
                    feed_dict['filename'] = img_fn

                    if len(joint_pos) == 16:
                        data = {
                            'filename': img_fn,
                            'train': train_flag,
                            'head_rect': head_rect,
                            'is_visible': vis,
                            'joint_pos': joint_pos
                        }

            # print(data)

            label = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']
            sss = ' '
            for key in label:
                sss = sss + str(int(data['joint_pos'][key][0])) + ' ' + str(int(data['joint_pos'][key][1])) + ' ' + str(
                    int(data['is_visible'][key])) + ' '
            sss = sss.strip()

            filename_set.setdefault(data['filename'], []).append(sss)
            # fout.write(data['filename'] + ' ' + sss + '\n')
    fout.close()


class MPII(Dataset):
    def __init__(self, root):
        self.root = root
        self.anno = []
        with open(os.path.join(root, "mpii_human_pose_v1_u12_2/mpii_list.txt")) as f:
            for line in f:
                data = line.strip().split()
                file, points = data[0], data[1:]
                anno = np.int64(points).reshape(16, 3)
                image = cv2.imread(os.path.join(root, "images", file))
                for i, key in enumerate(anno):
                    cv2.circle(image, (key[0], key[1]), 8, (0, 0, 255), 2)
                    cv2.putText(image, str(i), (key[0], key[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('src', image)
                cv2.waitKey(0)



if __name__ == '__main__':
    save_joints()

    # MPII(r"D:\temp_data\MPII")