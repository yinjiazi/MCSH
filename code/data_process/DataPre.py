# coding: utf-8
import atexit
import os
import argparse
import pandas as pd
from glob import glob
from tqdm import tqdm
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1


class dataPre():
    def __init__(self, working_dir):

        current_path = os.path.abspath('./')
        # self.working_dir = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
        self.working_dir = '/data1/yinjiazi/mmsa'

    def FetchFrames(self, input_dir, output_dir):  # FetchFrames('Raw', 'Processed/video/Frames')
        """
        fetch frames from raw videos using ffmpeg toolkits
        """
        # print(self.working_dir)  #/home/xjnu1/mmsa
        print("Start Fetch Frames...")
        # print(os.path.join(self.working_dir, input_dir)) #/home/xjnu1/mmsa/Raw
        # video_pathes = sorted(glob(os.path.join(self.working_dir, input_dir, '*/*.mp4')))
        video_pathes = sorted(glob(os.path.join(self.working_dir, input_dir, '*.mp4')))
        # print(video_pathes)  #/home/xjnu1/mmsa/Raw/10_001.mp4
        # print("---" * 20)
        output_dir = os.path.join(self.working_dir, output_dir)  # /data1/yinjiazi/mmsa/Processed/video/Frames

        # # test
        # video_path = '/data1/yinjiazi/mmsa/Raw/10_001.mp4'
        # out_path = '/data1/yinjiazi/mmsa/Processed/video/Frames/10_001/'
        # cmd = '/home/admin123/miniconda3/envs/yinjiazi/bin/ffmpeg -i ' + video_path + ' -r 10 ' + out_path + '\%04d.png'
        # os.system(cmd)

        for video_path in tqdm(video_pathes):
            # video_id, clip_id = video_path.split('\\')[-2:]
            video_id, clip_id = video_path.split('/')[-2:]
            clip_id = clip_id.split('.')[0]
            # clip_id = '%04d' % (int(clip_id))

            # e.g /home/xjnu1/mmsa/Processed/video/Frames/Raw/10_001
            cur_output_dir = os.path.join(output_dir, video_id, clip_id)

            if os.path.exists(cur_output_dir):
                continue
            if not os.path.exists(cur_output_dir):
                os.makedirs(cur_output_dir)
            cmd = '/home/admin123/miniconda3/envs/yinjiazi/bin/ffmpeg -i ' + video_path + ' -r 10 ' + cur_output_dir + '/' + '\%04d.png'
            os.system(cmd)

    def AlignFaces(self, input_dir, output_dir):  # ('Processed/video/Frames', 'Processed/video/AlignedFaces')
        """
        fetch faces from frames using MTCNN
        """
        import threading
        print("Start Align Faces...")

        def multi(frames_pathes):

            mtcnn = MTCNN(image_size=224, margin=0)
            for frames_path in tqdm(frames_pathes):

                output_path = frames_path.replace(input_dir, output_dir)
                # 忽略已经有的图片
                if os.path.exists(output_path):
                    continue

                if not os.path.exists(os.path.dirname(output_path)):
                    os.makedirs(os.path.dirname(output_path))

                # handle picture and save
                img = Image.open(frames_path)
                mtcnn(img, save_path=output_path)

        frames_pathes = sorted(glob(os.path.join(self.working_dir, input_dir, "*/*/*.png")))
        # dir_names = os.listdir(os.path.join(self.working_dir, input_dir,'1/'))
        length = len(frames_pathes)  # 总长
        n = 4  # 切分成多少份
        step = int(length / n) + 1  # 每份的长度
        container = []
        for i in range(0, length, step):
            split = frames_pathes[i: i + step]
            container.append(split)

        for i in range(n):
            thread = threading.Thread(target=multi, args=(container[i],))
            thread.start()
            # thread.join()

    def FetchAudios(self, input_dir, output_dir):
        """
        fetch audios from videos using ffmpeg toolkits
        """
        print("Start Fetch Audios...")
        # video_pathes = sorted(glob(os.path.join(self.working_dir, input_dir, '*.mp4')))
        video_pathes = sorted(glob(os.path.join(input_dir, '*.mp4')))
        for video_path in tqdm(video_pathes):
            output_path = video_path.replace(input_dir, output_dir).replace('.mp4', '.wav')
            if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path))
            # 调用ffmpeg执行音频提取功能
            # cmd = '/home/admin123/miniconda3/envs/yinjiazi/bin/ffmpeg -i ' + video_path + \
            #       ' -f wav -vn ' + output_path + ' -loglevel quiet'
            cmd = 'ffmpeg -i ' + video_path + ' -f wav -vn ' + output_path + ' -loglevel quiet'
            os.system(cmd)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='path to CH-SIMS')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    dp = dataPre(args.data_dir)
    print('-----begin-----')
    # print(args.data_dir)

    # fetch frames from videos
    # dp.FetchFrames('Raw', 'Processed/video/Frames')

    # align faces
    dp.AlignFaces('Processed/video/Frames', 'Processed/video/AlignedFaces')

    # fetch audio
    # dp.FetchAudios('Raw', 'Processed/audio') # run in 3 down