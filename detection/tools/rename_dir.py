from glob import glob
import os

def renumber(root):
    files = glob(os.path.join(root, "*"))
    prefix = os.path.basename(root)

    for i, file in enumerate(files):
        token = file.rsplit(".", 1)[-1]
        os.rename(file, os.path.join(root, f"{prefix}_{i}.{token}"))
        print(f"{file} rename to {os.path.join(root, f'{prefix}_{i}.{token}')}")

if __name__ == '__main__':
    renumber(r"E:\Data\OCR\my_code\photo\correct\SiamCAR")