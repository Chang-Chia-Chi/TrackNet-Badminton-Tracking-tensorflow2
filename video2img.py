import os
import cv2
import glob
import numpy as np
from parser import parser
from utils import genHeatMap

args = parser.parse_args()
HEIGHT=args.HEIGHT
WIDTH=args.WIDTH
sigma = args.sigma
mag = args.mag

def video2img(video, csv, output_path, match):
    """
    Convert videos to images in .jpg format

    param:
    video --> path of video to be convert
    csv --> path of csv recording position of ball
    output_path --> output path of images
    match --> index of match
    """
    with open(csv, 'r') as file:
        lines = file.readlines()[1:]

        csv_content = []
        for line in lines:
            frame, vis, x, y = line.strip().split(',')
            csv_content.append((int(frame), int(vis), float(x), float(y)))

    name_split = os.path.split(video)
    name = "match%d"%(match) + '_' + name_split[-1][:-4]

    count = 0
    num_data = len(csv_content)
    cap = cv2.VideoCapture(video)
    success, image = cap.read()
    ratio = image.shape[0]/HEIGHT
    while success:
        if count >= num_data:
            break
        label = csv_content[count]
        if label[1] == 0:
            heat_map = genHeatMap(WIDTH, HEIGHT, -1, -1, sigma, mag)
        else:
            heat_map = genHeatMap(WIDTH, HEIGHT, int(label[2]/ratio), int(label[3]/ratio), sigma, mag)
        
        image = cv2.resize(image, (WIDTH, HEIGHT))
        heat_map = (heat_map*255).astype('uint8')
        cv2.imwrite(os.sep.join([output_path, 'x_data', name+'_%d.jpg' %(count)]), image)
        cv2.imwrite(os.sep.join([output_path, 'y_data', name+'_%d.jpg' %(count)]), heat_map)
        success, image = cap.read()
        count += 1

if __name__ == '__main__':
    raw_dir = 'raw_data'
    videos = sorted(glob.glob(os.path.join(raw_dir, '*.mp4')))
    csvs = sorted(glob.glob(os.path.join(raw_dir, '*.csv')))

    match = 1
    print("==========Convert Start==========")
    for video, csv in zip(videos, csvs):
        v_name = os.path.split(video)[-1]
        csv_name = os.path.split(csv)[-1]
        if v_name[:-4] != csv_name[:-4]:
            raise NameError("Video files and csv files are not corresponded")

        print("Convert Video: {}".format(video))
        output_path = 'train_data/match%d'%(match)
        if not os.path.isdir(output_path):
            os.mkdir(output_path)
            os.mkdir(os.sep.join([output_path, 'x_data']))
            os.mkdir(os.sep.join([output_path, 'y_data']))
        video2img(video, csv, output_path, match)
        match += 1
        print("==========Convert End==========")