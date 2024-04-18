import os, warnings
import cv2
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from decord import VideoReader
from moviepy.editor import AudioFileClip

from scipy.io import wavfile # scipy library to read wav files
import numpy as np
from scipy.fftpack import dct
from matplotlib import pyplot as plt
from PIL import Image

input_size = 224
num_frame = 16
sampling_rate = 1

def read_video(file_path):
    vr = VideoReader(file_path)
    frames = vr.get_batch(range(len(vr))).asnumpy()
    return format_frames(
        frames,
        output_size=(input_size, input_size)
    )

def format_frames(frame, output_size):
    frame = tf.image.convert_image_dtype(frame, tf.uint8)
    frame = tf.image.resize(frame, size=list(output_size))
    return frame

def uniform_temporal_subsample(
    x, num_samples, clip_idx, total_clips, frame_rate=1, temporal_dim=-4
):
    t = tf.shape(x)[temporal_dim]
    max_offset = t - num_samples * frame_rate
    step = max_offset // total_clips
    offset = clip_idx * step
    indices = tf.linspace(
        tf.cast(offset, tf.float32),
        tf.cast(offset + (num_samples-1) * frame_rate, tf.float32),
        num_samples
    )
    indices = tf.clip_by_value(indices, 0, tf.cast(t - 1, tf.float32))
    indices = tf.cast(tf.round(indices), tf.int32)
    return tf.gather(x, indices, axis=temporal_dim)


def clip_generator(
    image, num_frames=32, frame_rate=1, num_clips=1, crop_size=224
):
    clips_list = []
    for i in range(num_clips):
        frame = uniform_temporal_subsample(
            image, num_frames, i, num_clips, frame_rate=frame_rate, temporal_dim=0
        )
        clips_list.append(frame)

    video = tf.stack(clips_list)
    video = tf.reshape(
        video, [num_clips*num_frames, crop_size, crop_size, 3]
    )
    return video

def video_audio(path, save_path):
    n = 1

    for class_name in os.listdir(path):
        class_dir = os.path.join(path, class_name)
        save_dir = os.path.join(save_path, class_name)
        
        for video_file in os.listdir(class_dir):
            video_path = os.path.join(class_dir, video_file)

            video_name = os.path.basename(video_path).split(".")[0]
            mp4_name = str(video_name) +  '.mp4'
            path_video_save = os.path.join(save_dir, mp4_name)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_video = cv2.VideoWriter(path_video_save, fourcc, 16.0, (224, 224))

            video_ds = read_video(video_path)
            video_ds = clip_generator(video_ds, num_frame, sampling_rate, num_clips=1)

            L1 = random.sample(range(0, 16), 16)
        # Random Face and Spectrogram
            for i in L1:
                video_img = video_ds.numpy()[i]
                video_img = video_img.astype('uint8')
                plt.axis('off')

                cv2.imwrite('video_img.jpg',video_img)
                video_img = Image.open("video_img.jpg")
                video_img = video_img.resize((224, 224))
                video_img = np.array(video_img)

                plt.close('all')
                output_video.write(video_img)

            output_video.release()
            cv2.destroyAllWindows()
            n = n + 1

    return n

if __name__ == '__main__':

    path_train = 'Data\\MEAD\\MEAD_OFOS\\train'
    save_train_path = 'Data\\MEAD\\MEAD_RFAS\\train'

    path_test = 'Data\\MEAD\\MEAD_OFOS\\test'
    save_test_path = 'Data\\MEAD\\MEAD_RFAS\\test'

    path_val = 'Data\\MEAD\\MEAD_OFOS\\val'
    save_val_path = 'Data\\MEAD\\MEAD_RFAS\\val'

    n_train = video_audio(path_train, save_train_path)
    n_test = video_audio(path_test, save_test_path)
    n_val = video_audio(path_val, save_val_path)

    print(n_train)
    print(n_test)
    print(n_val)
