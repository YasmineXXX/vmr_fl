import os
from tqdm import tqdm
charades_fps = 24

if __name__ == '__main__':
    original_annotation_path = ("./charades_sta_train.txt")
    end_list = []
    for x in open(original_annotation_path):
        video_begin_end, query = x.strip().split("##")
        video_name, clip_begin_second, clip_end_second = video_begin_end.split()
        end_list.append(clip_end_second)
    max_end = max(end_list)
    print(max_end)