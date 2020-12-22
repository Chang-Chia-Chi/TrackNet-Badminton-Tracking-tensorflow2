import os
import cv2
import csv
import random
import pickle
import numpy as np
from glob import glob
from functools import reduce
from collections import defaultdict

def genHeatMap(w, h, cx, cy, r, mag):
    """
    generate heat map of tracking badminton

    param:
    w: width of output heat map 
    h: height of output heat map
    cx: x coordinate of badminton
    cy: y coordinate of badminton
    r: radius of circle generated
    mag: factor to change range of grayscale
    """
    if cx == -1 or cy == -1:
        return np.zeros((h, w))

    x, y = np.meshgrid(np.linspace(1, w, w), np.linspace(1, h, h))
    heatmap = ((y - (cy + 1))**2) + ((x - (cx + 1))**2)
    heatmap[heatmap <= r**2] = 1
    heatmap[heatmap > r**2] = 0
    return heatmap*mag

def split_train_test(match_list, ratio=0.9, shuffle=True):
    """
    Split dataset into training and testing based on match list

    param:
    match_list --> list of match folder path
    ratio --> split ratio
    shuffle --> boolean to indicate whether to shuffle match_list 
                before generating dataset lists
    """
    if shuffle:
        random.shuffle(match_list)
        
    n_match = len(match_list)
    train_match = match_list[:int(n_match*ratio)]
    test_match = match_list[int(n_match*ratio):]
    x_train, y_train = [], []
    for match in train_match:
        train_imgs = glob(os.path.join(match, 'x_data', '*.jpg'))
        train_hmaps = glob(os.path.join(match, 'y_data', '*.jpg'))
        x_train.extend(train_imgs)
        y_train.extend(train_hmaps)

    x_test, y_test = [], []
    for match in test_match:
        test_imgs = glob(os.path.join(match, 'x_data', '*.jpg'))
        test_hmaps = glob(os.path.join(match, 'y_data', '*.jpg'))
        x_test.extend(test_imgs)
        y_test.extend(test_hmaps)

    return x_train, x_test, y_train, y_test

def read_img(file, hmap=False):
    """
    Read image from path and convert to format suitable for model
    
    param:
    file --> path of image file
    hmap --> boolean to indicate whether image is heat map or not
    """
    img = cv2.imread(file)
    if hmap:
        img = img[:,:,0]
        img = np.expand_dims(img, 0)
        return img.astype('float')/255.

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.expand_dims(img, 0)
    return img.astype('float')/255.
	
def check_steps(img_paths, batch_size, frame_stack):
    """
    Compute how many steps required for an training epoch

    param:
    img_paths --> list of image path
    batch_size --> batch size
    frame_stack --> number of frames to stack for one input
    """
    frame_counts = defaultdict(lambda: 0)
    for path in img_paths:
        video_name = reduce(lambda x, y:x+y, path.split('_')[:-1])
        frame_counts[video_name] += 1

    n_steps = 0
    for count in frame_counts.values():
        n_steps += (count - (frame_stack-1))//batch_size
    
    return n_steps - 1

def data_generator(batch_size, x_list, y_list, frame_stack):
    """
    Custom data generator to stack n frames for 'one' input

    param:
    batch_size --> batch size
    x_list --> image path list
    y_list --> heat map path list
    frame_stack --> number of frames to stack for one input
    """
    x_list = sorted(x_list)
    y_list = sorted(y_list)
    data_size = len(x_list)

	# initialize images and heatmaps array
    END = False
    end = (frame_stack-1) + (batch_size-1)
    images = [read_img(path) for path in x_list[:frame_stack]]
    hmap = read_img(y_list[frame_stack-1], hmap=True)
    while True:
        batch_imgs = []
        batch_hmaps = []
		
		# dynamically pop and append a new image to avoid multiple reading
        for i in reversed(range(batch_size)):
            img = np.concatenate(images, axis=0)
            batch_imgs.append(img)
            images.pop(0)
            images.append(read_img(x_list[end]))

            batch_hmaps.append(hmap)
            hmap = read_img(y_list[end], hmap=True)
			
            end += 1
            if end >= data_size:
                END = True
                break
			
			# if image comes from different video, reset images and heat_maps
            next_info = os.path.split(x_list[end])[-1].split('_')
            curr_info = os.path.split(x_list[end-1])[-1].split('_')
            if next_info[:-1] != curr_info[:-1]:
                images = [read_img(path) for path in x_list[end:end+frame_stack]]
                heat_maps = read_img(y_list[end+(frame_stack-1)], hmap=True)
                end += frame_stack
                break
        if END:
            END=False
            end = (frame_stack-1) + (batch_size-1)
            images = [read_img(path) for path in x_list[:frame_stack]]
            hmap = read_img(y_list[frame_stack-1], hmap=True)
            continue
        
        yield np.array(batch_imgs), np.array(batch_hmaps)

def confusion(y_pred, y_true, tol):
    """
    compute confusion matrix value
    TP: True positive
    TN: True negative
    FP2: False positive
    FN: False negative
    FP1: If distance of ball center between 
         ground truth and prediction is larger than tolerance

    param:
    y_pred --> predicted heat map
    y_true --> ground truth heat map
    tol --> acceptable tolerance of heat map circle center 
            between ground truth and prediction
    """
    
    batch_size = y_pred.shape[0]
    TP = TN = FP1 = FP2 = FN = 0
    for b in range(batch_size):
        h_pred = y_pred[b]*255
        h_pred = h_pred.astype('uint8')
        h_true = y_true[b]*255
        h_true = h_true.astype('uint8')
        if np.amax(h_pred)==0 and np.amax(h_true)==0:
            TN += 1
        elif np.amax(h_pred)>0 and np.amax(h_true)==0:
            FP2 += 1
        elif np.amax(h_pred)==0 and np.amax(h_true)>0:
            FN += 1
        elif np.amax(h_pred)>0 and np.amax(h_true)>0:
            # find center of ball for prediction
            _, contours, _ = cv2.findContours(h_pred[0].copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            bboxes = [cv2.boundingRect(cnt) for cnt in contours]

            areas = np.array([bbox[2] * bbox[3] for bbox in bboxes])
            target = bboxes[np.argmax(areas)]
            x, y, w, h = target
            (cx_pred, cy_pred) = (int(x+w/2), int(y+h/2))

            # find center of ball for ground truth
            _, contours, _ = cv2.findContours(h_true.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            bboxes = [cv2.boundingRect(cnt) for cnt in contours]

            areas = np.array([bbox[2] * bbox[3] for bbox in bboxes])
            target = bboxes[np.argmax(areas)]
            x, y, w, h = target
            (cx_true, cy_true) = (int(x+w/2), int(y+h/2))

            dist = ((cx_pred-cx_true)**2 + (cy_pred-cy_true)**2)**0.5
            if dist > tol:
                FP1 += 1
            else:
                TP += 1
    
    return (TP, TN, FP1, FP2, FN)

def compute_acc(evaluation):
    """
    Compute accuracy, precision and recall

    parame:
    evaluation --> a tuple containing 5 variable(TP, TN, FP1, FP2, FN)
    """
    (TP, TN, FP1, FP2, FN) = evaluation
    try:
	    accuracy = (TP + TN) / (TP + TN + FP1 + FP2 + FN)
    except:
	    accuracy = 0
    try:
        precision = TP / (TP + FP1 + FP2)
    except:
        precision = 0
    try:
        recall = TP / (TP + FN)
    except:
        recall = 0
    return accuracy, precision, recall

def save_info(info, video_path):
    success = False
    try:
        video_name = os.path.split(video_path)[-1][:-4]
        with open(video_name+'.csv', 'w') as file:
            file.write("Frame,Ball,x,y\n")
            for frame in info:
                data = "{},{},{:.3f},{:.3f}".format(info[frame]["Frame"], info[frame]["Ball"],
                                            info[frame]["x"],info[frame]["y"])
                file.write(data+'\n')
        success = True
        print("Save info successfully into", video_name+'.csv')
    except:
        print("Save info failure")

    return success

def load_info(csv_path):
    with open(csv_path, 'r') as file:
        lines = file.readlines()
        n_frames = len(lines) - 1
        info = {
            idx:{
            'Frame': idx,
            'Ball': 0,
            'x': -1,
            'y': -1
            } for idx in range(n_frames)
        }

        for line in lines[1:]:
            frame, ball, x, y = line.split(',')
            frame = int(frame)
            info[frame]['Frame'] = frame
            info[frame]['Ball'] = int(ball)
            info[frame]['x'] = float(x)
            info[frame]['y'] = float(y)

    return info

def show_image(image, frame_no, x, y):
    h, w, _ = image.shape
    if x != -1 and y != -1:
        x_pos = int(x*w)
        y_pos = int(y*h)
        cv2.circle(image, (x_pos, y_pos), 5, (0, 0, 255), -1)
    text = "Frame: {}".format(frame_no)
    cv2.putText(image, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv2.LINE_AA)
    return image

def go2frame(cap, frame_no, info):
    x, y = info[frame_no]['x'], info[frame_no]['y']
    cap.set(1, frame_no)
    ret, image = cap.read()
    image = show_image(image, frame_no, x, y)
    return image