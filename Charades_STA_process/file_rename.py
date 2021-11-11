import os

frame_path = "./Charades_v1_rgb/"

if __name__ == '__main__':
    n = 0
    for video_dir in os.listdir(frame_path):
        #print(video_dir)
        video_dir = os.path.join(frame_path, video_dir)
        for file in os.listdir(video_dir):
            video_name, label = file.split('-')
            new_name = file.replace(file, "img_%s"%label)
            os.rename(os.path.join(video_dir, file), os.path.join(video_dir, new_name))
        n += 1
        print("%d video has done"%n)
