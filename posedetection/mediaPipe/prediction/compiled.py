import sys
import getopt
import numpy as np
import os
import queue
import csv
from glob import glob
import piexif
from keras_preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import *
from keras.layers import *
from TrackNet3 import TrackNet3
import keras.backend as K
from keras import optimizers
import tensorflow as tf
import cv2
from os.path import isfile, join
from PIL import Image, ImageDraw
import time
import mediapipe as mp
import pickle

# For GUI  library (Input UI)
import tkinter.filedialog as filedialog
import tkinter as tk

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
BATCH_SIZE=1
HEIGHT=288
WIDTH=512
#HEIGHT=360
#WIDTH=640
sigma=2.5
mag=1

def genHeatMap(w, h, cx, cy, r, mag):
	if cx < 0 or cy < 0:
		return np.zeros((h, w))
	x, y = np.meshgrid(np.linspace(1, w, w), np.linspace(1, h, h))
	heatmap = ((y - (cy + 1))**2) + ((x - (cx + 1))**2)
	heatmap[heatmap <= r**2] = 1
	heatmap[heatmap > r**2] = 0
	return heatmap*mag

#time: in milliseconds
def custom_time(time):
	remain = int(time / 1000)
	ms = (time / 1000) - remain
	s = remain % 60
	s += ms
	remain = int(remain / 60)
	m = remain % 60
	remain = int(remain / 60)
	h = remain
	#Generate custom time string
	cts = ''
	if len(str(h)) >= 2:
		cts += str(h)
	else:
		for i in range(2 - len(str(h))):
			cts += '0'
		cts += str(h)
	
	cts += ':'

	if len(str(m)) >= 2:
		cts += str(m)
	else:
		for i in range(2 - len(str(m))):
			cts += '0'
		cts += str(m)

	cts += ':'

	if len(str(int(s))) == 1:
		cts += '0'
	cts += str(s)

	return cts

def custom_loss(y_true, y_pred):
	loss = (-1)*(K.square(1 - y_pred) * y_true * K.log(K.clip(y_pred, K.epsilon(), 1)) + K.square(y_pred) * (1 - y_true) * K.log(K.clip(1 - y_pred, K.epsilon(), 1)))
	return K.mean(loss)

def calculateAngle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radian = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radian*180.0/np.pi)

    if angle > 180.0:
        angle = 360-angle

    return angle

load_weights = "model906_30"
classifier = "modelv01.pkl"

def trackNetPhase1(videoToCapture, filename):
    videoName = filename
    video = videoToCapture[0]
    model = load_model(load_weights, custom_objects={'custom_loss':custom_loss})
    model.summary()
    print('Beginning predicting......')

    start = time.time()
    # f = open("test.csv", "w"c)
    f = open(videoName[:-4]+"_predict.csv", "w")
    f.write('Frame,Visibility,X,Y,Time\n')


    cap = cv2.VideoCapture(videoName)

    success, image1 = cap.read()
    frame_time1 = custom_time(cap.get(cv2.CAP_PROP_POS_MSEC))
    success, image2 = cap.read()
    frame_time2 = custom_time(cap.get(cv2.CAP_PROP_POS_MSEC))
    success, image3 = cap.read()
    frame_time3 = custom_time(cap.get(cv2.CAP_PROP_POS_MSEC))

    ratio = image1.shape[0] / HEIGHT

    size = (int(WIDTH*ratio), int(HEIGHT*ratio))
    fps = 30

    if videoName[-3:] == 'avi':
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    elif videoName[-3:] == 'mp4':
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    else:
        print('usage: video type can only be .avi or .mp4')
        exit(1)

    out = cv2.VideoWriter(videoName[:-4]+'_predict'+videoName[-4:], fourcc, fps, size)

    count = 0

    while success:
        unit = []
        #Adjust BGR format (cv2) to RGB format (PIL)
        x1 = image1[...,::-1]
        x2 = image2[...,::-1]
        x3 = image3[...,::-1]
        #Convert np arrays to PIL images
        x1 = array_to_img(x1)
        x2 = array_to_img(x2)
        x3 = array_to_img(x3)
        #Resize the images
        x1 = x1.resize(size = (WIDTH, HEIGHT))
        x2 = x2.resize(size = (WIDTH, HEIGHT))
        x3 = x3.resize(size = (WIDTH, HEIGHT))
        #Convert images to np arrays and adjust to channels first
        x1 = np.moveaxis(img_to_array(x1), -1, 0)
        x2 = np.moveaxis(img_to_array(x2), -1, 0)
        x3 = np.moveaxis(img_to_array(x3), -1, 0)
        #Create data
        unit.append(x1[0])
        unit.append(x1[1])
        unit.append(x1[2])
        unit.append(x2[0])
        unit.append(x2[1])
        unit.append(x2[2])
        unit.append(x3[0])
        unit.append(x3[1])
        unit.append(x3[2])
        unit=np.asarray(unit)	
        unit = unit.reshape((1, 9, HEIGHT, WIDTH))
        unit = unit.astype('float32')
        unit /= 255
        y_pred = model.predict(unit, batch_size=BATCH_SIZE)
        y_pred = y_pred > 0.5
        y_pred = y_pred.astype('float32')
        h_pred = y_pred[0]*255
        h_pred = h_pred.astype('uint8')
        for i in range(3):
            if i == 0:
                frame_time = frame_time1
                image = image1
            elif i == 1:
                frame_time = frame_time2
                image = image2
            elif i == 2:
                frame_time = frame_time3	
                image = image3

            if np.amax(h_pred[i]) <= 0:
                f.write(str(count)+',0,0,0,'+frame_time+'\n')
                out.write(image)
            else:	
                #h_pred
                (cnts, _) = cv2.findContours(h_pred[i].copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                rects = [cv2.boundingRect(ctr) for ctr in cnts]
                max_area_idx = 0
                max_area = rects[max_area_idx][2] * rects[max_area_idx][3]
                for i in range(len(rects)):
                    area = rects[i][2] * rects[i][3]
                    if area > max_area:
                        max_area_idx = i
                        max_area = area
                target = rects[max_area_idx]
                (cx_pred, cy_pred) = (int(ratio*(target[0] + target[2] / 2)), int(ratio*(target[1] + target[3] / 2)))

                f.write(str(count)+',1,'+str(cx_pred)+','+str(cy_pred)+','+frame_time+'\n')
                image_cp = np.copy(image)
                cv2.circle(image_cp, (cx_pred, cy_pred), 5, (0,0,255), -1)
                out.write(image_cp)
            count += 1
        success, image1 = cap.read()
        frame_time1 = custom_time(cap.get(cv2.CAP_PROP_POS_MSEC))
        success, image2 = cap.read()
        frame_time2 = custom_time(cap.get(cv2.CAP_PROP_POS_MSEC))
        success, image3 = cap.read()
        frame_time3 = custom_time(cap.get(cv2.CAP_PROP_POS_MSEC))

    f.close()
    out.release()
    end = time.time()
    print('Prediction time:', end-start, 'secs')
    print('Done......')

def trackNetPhase2(videoToCapture, filename):
    input_csv_path = videoToCapture[0] +"_predict.csv"
    input_video_path = videoToCapture[0] +"_predict.mp4"

    with open(input_csv_path) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        frames = []
        x, y = [], []
        list1 = []
        for row in readCSV:
            list1.append(row)
        for i in range(1 , len(list1)):
            frames += [int(list1[i][0])]
            x += [int(float(list1[i][2]))]
            y += [int(float(list1[i][3]))]

    output_video_path = input_video_path.split('.')[0] + "_trajectory.mp4"

    q = queue.deque()
    for i in range(0,8):
        q.appendleft(None)

    #get video fps&video size
    currentFrame= 0
    video = cv2.VideoCapture(input_video_path)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    output_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    output_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video = cv2.VideoWriter(output_video_path,fourcc, fps, (output_width,output_height))

    video.set(1,currentFrame); 
    ret, img1 = video.read()
    #write image to video
    output_video.write(img1)
    currentFrame +=1
    #input must be float type
    img1 = img1.astype(np.float32)

    #capture frame-by-frame
    video.set(1,currentFrame);
    ret, img = video.read()
    #write image to video
    output_video.write(img)
    currentFrame +=1
    #input must be float type
    img = img.astype(np.float32)

    while(True):

        #capture frame-by-frame
        video.set(1,currentFrame); 
        ret, img = video.read()
            #if there dont have any frame in video, break
        if not ret: 
            break
        PIL_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   
        PIL_image = Image.fromarray(PIL_image)

        if x[currentFrame] != 0 and y[currentFrame] != 0:
            q.appendleft([x[currentFrame],y[currentFrame]])
            q.pop()
        else:
            q.appendleft(None)
            q.pop()

        for i in range(0,8):
            if q[i] is not None:
                draw_x = q[i][0]
                draw_y = q[i][1]
                bbox =  (draw_x - 2, draw_y - 2, draw_x + 2, draw_y + 2)
                draw = ImageDraw.Draw(PIL_image)
                draw.ellipse(bbox, outline ='yellow')
                del draw
        opencvImage =  cv2.cvtColor(np.array(PIL_image), cv2.COLOR_RGB2BGR)
        #write image to output_video
        output_video.write(opencvImage)

        #next frame
        currentFrame += 1

    video.release()
    output_video.release()
    print("finish")
    return output_video_path


############## Mediapipe prediction ################
with open(classifier, 'rb') as f:
    model = pickle.load(f)


def analyse(videoToCapture, output_directory, isLeftHanded = False):
    # read video
    cap = cv2.VideoCapture(videoToCapture)

    # information for output
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    output_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    output_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    new_output = output_directory + videoToCapture.split('/')[-1].split('.')[0] + "_output.mp4"
    output_video = cv2.VideoWriter(new_output, fourcc, fps, (output_width, output_height))
    crop_rate = 5
    # Mediapipe Instance
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        last_pose = "Unknown"
        #simulate tracking shuttlecock
        print("Tracking shuttlecock...")
        print("Shuttlecock tracking done")
        print("Processing shuttlecock trajectory...")
        print("Shuttlecock trajectory processed")
        print("Preview window is open")
        print("Analysing shot styles... Press 'q' to quit")
        while True:
            ret, frame = cap.read()

            if ret:
                # Recoloring Image
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Make Detection
                results = pose.process(image)

                # Recolor back
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Extract

                try:
                    # Check if is left handed
                    if isLeftHanded:
                        cv2.flip(image, 1)

                    # Retrieve all landmarks
                    landmarks = results.pose_landmarks.landmark
                    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

                    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]

                    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

                    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

                    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                    # calculate all angles
                    left_arm = calculateAngle(left_shoulder, left_elbow, left_wrist)
                    right_arm = calculateAngle(right_shoulder, right_elbow, right_wrist)

                    left_leg = calculateAngle(left_hip, left_knee, left_ankle)
                    right_leg = calculateAngle(right_hip, right_knee, right_ankle)

                    left_under_arm = calculateAngle(left_elbow, left_shoulder, left_hip)
                    right_under_arm = calculateAngle(right_elbow, right_shoulder, right_hip)

                    # Check if is left handed
                    if isLeftHanded:
                        cv2.flip(image, 1)
                    
                    # Construct dataframe for prediction
                    lst = np.array(
                        [left_arm, right_arm, left_leg, right_leg, left_under_arm, right_under_arm]).flatten()
                    lst = list(lst)
                    x = pd.DataFrame([lst])
                    
                    # Predict the results
                    model_class = model.predict(x.values)[0]

                    # Get the confidence rate
                    model_prob = model.predict_proba(x.values)[0]

                    # check if confidence rate is > 50%
                    if (round(model_prob[np.argmax(model_prob)], 2) > 0.5):

                        # Display classified class
                        cv2.putText(image, model_class, [25, 25], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
                                    cv2.LINE_AA)
                    else:
                        # If confidence rate < 50, show unknown
                        cv2.putText(image, "Unknown", [25, 25], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
                                    cv2.LINE_AA)

                    # Display Probability
                    cv2.putText(image, str(round(model_prob[np.argmax(model_prob)], 2)), [35, 45],
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                except:
                    pass

                # Render Detection
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # draw preview and write to output
                output_video.write(image)
                cv2.imshow('MediaPipe', image)

                # use Q to terminate
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            else:
                break
        # terminate program
        output_video.release()
        cap.release()
        cv2.destroyAllWindows()


############### INPUT GUI ###############


screen = tk.Tk()
screen.title('BadminSmash')  # Set the title of GUI window

# input ouput path variables
filename = None
output_directory = None


def input():
    input_path = tk.filedialog.askopenfilename(filetypes=[("Video files", "*.mp4")])
    input_entry.delete(1, tk.END)  # Remove current text in entry
    input_entry.insert(0, input_path)  # Insert the 'input path'


def output():
    output_path = tk.filedialog.askdirectory()
    output_entry.delete(1, tk.END)  # Remove current text in entry
    output_entry.insert(0, output_path)  # Insert the 'output path'

# CheckBox for left hand analysis
Checkbutton1 = tk.IntVar()
leftHandButton = tk.Checkbutton(screen, text = "Left Handed", 
                      variable = Checkbutton1,
                      onvalue = 1,
                      offvalue = 0,
                      height = 2,
                      width = 15)


# begin button logic
def begin():
    leftHandBool = 0
    filename = input_entry.get()
    output_directory = output_entry.get()
    output_directory += "/"
    # print(filename)
    # print(output_directory)
    if filename != '' and output_directory != '':
        if Checkbutton1.get() == 1:
            leftHandBool = 1
        
        filenameBreakdown = filename.split('.')
        print("Tracking Shuttlecock...")
        trackNetPhase1(filenameBreakdown, filename)
        print("Shuttlecock tracking done")
        print("Processing shuttlecock trajectory...")
        fileLocAfterTrackNet = trackNetPhase2(filenameBreakdown, filename)
        print("Shuttlecock trajectory processed")
        print("Analysing shot styles... Press 'q' to quit the video preview window.")
        analyse(fileLocAfterTrackNet, output_directory, leftHandBool)  # start mediapipe
        print("Analysis done. Output video saved.")



# HEADER TEXT
text = '''Welcome! This software was developed by MCS13. \n
You may use this programme to analyse badminton match videos for the following shots:
    - Serve
    - Forehand
    - Backhand
    - Net Drop \n
        
            Please enter the file paths below.
        Press 'q' to quit the video preview window.
        '''
# Tkinter text box
textBox = tk.Text(screen, height=13, width=60);
textBox.insert("end", text)
# To make it non-editable use the config function
textBox.config(state='disabled')

# Frame boxes
# header_frame = tk.Frame(screen)
top_frame = tk.Frame(screen)
bottom_frame = tk.Frame(screen)
line1 = tk.Frame(screen, height=1, width=400, bg="grey80", relief='groove')
line2 = tk.Frame(screen, height=1, width=400, bg="grey80", relief='groove')

# INPUT
input_path = tk.Label(top_frame, text="Input File Path:")
input_entry = tk.Entry(top_frame, text="", width=40)
browse1 = tk.Button(top_frame, text="Browse", command=input)

# OUTPUT
output_path = tk.Label(bottom_frame, text="Output File Path:")
output_entry = tk.Entry(bottom_frame, text="", width=40)
browse2 = tk.Button(bottom_frame, text="Browse", command=output)

# BUTTONS
begin_button = tk.Button(bottom_frame, text='Begin Analysis', command=begin)
end_button = tk.Button(bottom_frame, text='Quit', command=screen.quit)

textBox.pack(expand=True)
leftHandButton.pack(pady=5)
line1.pack(pady=10)
top_frame.pack()
line2.pack(pady=10)
bottom_frame.pack(side=tk.BOTTOM)
input_path.pack(pady=5)
input_entry.pack(pady=5)
browse1.pack(pady=5)
output_path.pack(pady=5)
output_entry.pack(pady=5)
browse2.pack(pady=5)

begin_button.pack(pady=7, fill=tk.X)
end_button.pack(pady=10, fill=tk.X)

# main instance for input GUI
screen.mainloop()