import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
import pickle

# For GUI  library (Input UI)
import tkinter.filedialog as filedialog
import tkinter as tk

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
BATCH_SIZE = 1
HEIGHT = 288
WIDTH = 512
# HEIGHT=360
# WIDTH=640
sigma = 2.5
mag = 1


def calculateAngle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radian = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radian * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


# classifier model to be used
classifier = "modelv01.pkl"

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
        analyse(filename, output_directory, leftHandBool)  # start mediapipe



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