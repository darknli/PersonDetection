import cv2
import os
from glob import glob
import random
import numpy as np
import copy
import json
from tqdm import tqdm


def padding_image(image, diff, axis):
    h, w = image.shape[:2]
    if diff > 0:
        if axis == 0:
            new_h = h + diff
            new_image = np.zeros((new_h, w, 3), dtype=image.dtype)
        elif axis == 1:
            new_w = w + diff
            new_image = np.zeros((h, new_w, 3), dtype=image.dtype)
        else:
            raise ValueError
        new_image[:, :] = (65, 63, 60)  # 置成灰色
        new_image[:h, :w] = image
        return new_image
    elif diff < 0:
        if axis == 0:
            return image[:h + diff].copy()
        elif axis == 1:
            return image[:, :w + diff].copy()
        else:
            raise ValueError
    return image


def get_item(label, x1, y1, x2, y2):
    item = {
        "label": label,
        "points": [
            [
                float(x1),
                float(y1)
            ],
            [
                float(x2),
                float(y2)
            ]
        ],
        "group_id": None,
        "shape_type": "rectangle",
        "flags": {}
    }
    return item


class Nesting:
    def __init__(self, root, code_root):
        self.main_space = cv2.imread(os.path.join(root, "main.png"))
        self.projects = glob(os.path.join(root, "projects/*.png"))
        self.clicked_tabs = glob(os.path.join(root, "tabs/clicked/*.png"))
        self.tabs = glob(os.path.join(root, "tabs/others/*.png"))
        self.title = glob(os.path.join(root, "title/*.png"))
        self.consoles = glob(os.path.join(root, "console*.png"))

        with open(os.path.join(root, "main.json")) as f:
            info = json.loads(f.read())["shapes"]
        self.ms_info = {box["label"]: np.array(box["points"], dtype=int).reshape(-1) for box in info}
        self.code_patches = self.get_code_patches(code_root)

    def get_code_patches(self, root):
        files = glob(os.path.join(root, "*.json"))
        pbar = tqdm(files)
        patches = []

        for file in pbar:
            with open(file, encoding="utf-8") as f:
                ori_code = json.loads(f.read())
            target_code = {
                "images": os.path.join(root, os.path.basename(ori_code["imagePath"])),
                "height": ori_code["imageHeight"],
                "width": ori_code["imageWidth"],
            }
            image = cv2.imread(target_code["images"])
            for sub_data in ori_code["shapes"]:
                if sub_data["label"] == "^":
                    label = "blank"
                    patch_image = 10  # 默认空行宽度
                    # if random.random() < 0.8:
                    #     continue
                elif sub_data["label"].startswith("^"):
                    x1, y1 = sub_data["points"][0]
                    x2, y2 = sub_data["points"][1]
                    patch_image = image[int(y1): int(y2), int(x1): int(x2)].copy()
                    if x2 - x1 < 10:
                        print(file, sub_data["label"])
                else:
                    continue
                box = {
                    "label": sub_data["label"],
                    "image": patch_image
                }
                patches.append(box)
        pbar.close()
        return patches

    def random_merge(self, dst_name, crop_pixel_idx=400, crop_body=True, imitate_screen=True):
        main_space = self.main_space.copy()
        slet_tab_name = random.choice(self.clicked_tabs)
        slet_tab = cv2.imread(slet_tab_name)
        slet_title = cv2.imread(random.choice(self.title))
        slet_project = cv2.imread(random.choice(self.projects))
        slet_console = cv2.imread(random.choice(self.consoles))
        tabs = copy.deepcopy(self.tabs)

        ms_h, ms_w = main_space.shape[:2]
        sp_h, sp_w = slet_project.shape[:2]
        st_h, st_w = slet_title.shape[:2]

        slet_project = padding_image(slet_project, ms_h - sp_h, 0)
        slet_console = padding_image(slet_console, st_w - slet_console.shape[1], 1)

        diff = ms_w + sp_w - st_w
        main_space = np.concatenate(
            [main_space[:, :-crop_pixel_idx - diff], main_space[:, -crop_pixel_idx:]], 1)

        num_tabs = random.randint(1, 12)
        tabs_list = []
        for _ in range(num_tabs):
            tab = random.choice(tabs)
            tab_image = cv2.imread(tab)
            tabs.remove(tab)
            tab_image = self.resize(tab_image, self.ms_info["title"][3])
            tabs_list.append(tab_image)
        np.random.shuffle(tabs_list)
        slet_tab = self.resize(slet_tab, self.ms_info["title"][3])
        tab_insert_idx = random.randint(0, num_tabs)
        tabs_list.insert(tab_insert_idx, slet_tab)
        clicked_start_w = sum([img.shape[1] for img in tabs_list[:tab_insert_idx]])
        clicked_end_w = clicked_start_w + slet_tab.shape[1]
        tot_tabs = np.concatenate(tabs_list, 1)

        h, w = tot_tabs.shape[:2]
        try:
            main_space[:h, :w] = tot_tabs
        except:
            print()

        span = 3
        w, h = self.ms_info["line"][2:] - self.ms_info["line"][:2] + 3
        x1 = self.ms_info["line"][0]
        y1 = self.ms_info["line"][1]
        # for i in range(30):
        #     print((self.ms_info["line"][0], height), (random.randint(300, 1000), height+h))
        #     cv2.rectangle(main_space, (self.ms_info["line"][0], height), (random.randint(300, 1000), height+h),
        #                   (0, 255, 0), 1, cv2.LINE_4)
        #     height += h + span

        boxes = []
        label_list = []
        for i in range(58):
            y2 = y1 + h
            box = random.choice(self.code_patches)
            label, image = box["label"], box["image"]
            if label == "^":
                # image = np.ones((h, image, 3), dtype=main_space.dtype)
                x2 = x1 + image
                pass
            else:
                # print(f"""{i + 1}'{label}', box={image.shape}""", end="-->")
                image = self.resize(image, h+4)
                w = image.shape[1]
                x2 = x1 + w
                # print(f"{x1, y1, x2, y2}")
                main_space[y1-2: y2+2, x1: x2] = image
            boxes.append([x1, y1-2, x2, y2+2])
            label_list.append(label)
            y1 = y2 + span

        boxes.append([clicked_start_w, 0, clicked_end_w, self.ms_info["title"][3]])
        label_list.append(slet_tab_name)

        boxes = np.array(boxes)
        main_space_bbox = np.array([0, 0, main_space.shape[1], main_space.shape[0]])

        if not crop_body:
            # tab位表头
            main_space = np.concatenate([slet_project, main_space], 1)
            boxes[:, ::2] += slet_project.shape[1]
            main_space_bbox[::2] += slet_project.shape[1]

            body = np.concatenate([
                slet_title,
                main_space,
                slet_console
            ], 0)
            boxes[:, 1::2] += slet_title.shape[0]
            main_space_bbox[1::2] += slet_title.shape[0]
        else:
            body = main_space

        # for box, label in zip(boxes, label_list):
        #     x1, y1, x2, y2 = box
        #     if label == "^":
        #         cv2.rectangle(body, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_4)
        #     elif not label.startswith("^"):
        #         cv2.rectangle(body, (x1, y1), (x2, y2), (255, 0, 0), 1, cv2.LINE_4)
        #     else:
        #         cv2.rectangle(body, (x1, y1), (x2, y2), (0, 255, 0), 1, cv2.LINE_4)
        # cv2.rectangle(body, (main_space_bbox[0], main_space_bbox[1]), (main_space_bbox[2], main_space_bbox[3]), (0, 255, 0), 1, cv2.LINE_4)
        # cv2.imshow("body", body)
        # cv2.waitKey()

        if imitate_screen:
            h, w = body.shape[:2]
            aug = np.random.normal(80, 21, [3, 3])
            aug = np.tile(aug[:, :, None], (1, 1, 3))
            aug = np.clip(aug, 0, 100)
            # aug[:, :, 1] += 10
            # aug[:, :, 2] += 15
            aug = cv2.resize(aug, (w // 10, h // 10))
            aug = cv2.GaussianBlur(aug, (3, 3), 3)
            aug = cv2.resize(aug, (w, h)).astype(int)
            body = np.clip(body.astype(int) + aug, 0, 255).astype(np.uint8)
            if random.random() < 0.5:
                size = random.randint(1, 2)
                sigma = random.randint(6, 15)
                k = size * 2 + 1
                body = cv2.GaussianBlur(body, (k, k), k/sigma)

        anno = {
            "version": "4.5.12",
            "flags": {},
            "imagePath": os.path.basename(dst_name),
            "imageHeight": 1248,
            "imageWidth": 2521,
            "main_space_bbox": main_space_bbox.tolist()
        }
        anno["shapes"] = [get_item(label, *box) for label, box in zip(label_list, boxes)]
        anno["area"] = []
        with open(dst_name.replace("png", "json"), "w") as f:
            json.dump(anno, f)
        cv2.imwrite(dst_name, body)

    def resize(self, image, target_h):
        h, w = image.shape[:2]
        image = cv2.resize(image, (int(w * target_h / h + 0.5), target_h))
        return image


if __name__ == '__main__':
    path = "D:/temp_data/code_data/template"
    code_root = "D:/temp_data/code_data/recognition/raw_images"
    n = Nesting(path, code_root)
    for i in range(700):
        n.random_merge(f"E:/Learn/OCR_Pytorch/obective_detection/augment_images/aug_{i}.png")
