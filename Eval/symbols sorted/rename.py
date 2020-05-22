import os

BASE_DIR = os.path.dirname(__file__)

for root, dirs, files in os.walk(BASE_DIR):
    count = 0
    for file in files:
        file = file.lower()
        if not file.endswith("py"):# or file.endswith("jpg") or file.endswith("jpeg"):
            ori_file_path = os.path.join(root, file)
            rn_file_path = os.path.join(root, os.path.split(root)[1] + '_' + str(count) + '.jpg')
            os.rename(ori_file_path, rn_file_path)
            count += 1