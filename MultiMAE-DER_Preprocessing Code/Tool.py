import os

def rename(read_path, save_path):
    n = 0

    for people_name in os.listdir(read_path):
        people_dir = os.path.join(read_path, people_name)

        for class_name in os.listdir(people_dir):
            class_dir = os.path.join(people_dir, class_name)
            save_dir = os.path.join(save_path, class_name)

            for level_name in os.listdir(class_dir):
                level_dir = os.path.join(class_dir, level_name)

                for video_file in os.listdir(level_dir):
                    video_path = os.path.join(level_dir, video_file)
                    video_name = os.path.basename(video_path).split(".")[0]

                    rename = str(people_name) + '_' + str(class_name) + '_' + str(level_name) + '_' + str(video_name) + '.mp4'
                    rename_path = os.path.join(save_dir, rename)

                    if os.path.exists(video_path):
                        os.rename(video_path, rename_path)
                        n = n + 1

    return n

if __name__ == '__main__':

    read_path = 'Data\\MEAD_RAW'
    save_path = 'Data\\MEAD'

    n = rename(read_path, save_path)
    print(n)
