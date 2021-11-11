import os
from tqdm import tqdm
charades_fps = 24

if __name__ == '__main__':
    original_annotation_path = ("./charades_sta_test.txt")
    original_file_path = "/data/wangyan/Charades-STA/Charades_v1_rgb/"
    target_file_path = "./charades_annotations_test.txt"
    target_file = open(target_file_path, mode='w')
    query_label = 0
    for x in open(original_annotation_path):
        video_begin_end, query = x.strip().split("##")
        video_name, clip_begin_second, clip_end_second = video_begin_end.split()
        clip_begin_frame = int(float(clip_begin_second) * charades_fps)
        clip_end_frame = int(float(clip_end_second) * charades_fps)
        original_video_path = os.path.join(original_file_path, video_name)
        video_end_frame = len(os.listdir(original_video_path))
        target_file.writelines([video_name, '_', '1_', str(video_end_frame), '_',
                                query, '_', str(clip_begin_frame), '_', str(clip_end_frame),
                                '_', str(clip_begin_second), '_', str(clip_end_second), '\n'])
        query_label += 1
    target_file.close()