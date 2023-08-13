import get_point_and_drawed_img ##for the first time after that make it comments

import os
import cv2
import mediapipe as mp
import numpy as np
from all_reference_poses_point_deletable import reference_pose
from config import round_pose, scale, cam_type
import needed_func as nf

#Initialize pose mediapipe and OpenCV
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

#set webcam max resolution
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,2000);  cap.set(cv2.CAP_PROP_FRAME_WIDTH,2000)

w_cam = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h_cam = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
win_cam = (int(w_cam),int(h_cam))
win_monitor = (int(w_cam*scale),int(h_cam*scale))

#get drawed image in folder
image_drawed = "Drawed reference image deletable"
filename = os.listdir(image_drawed)

#random image 
image_passed = set()
while True:
        random_image = np.random.randint(0, len(filename))
        image_passed.add(random_image)
        if len(image_passed) >= round_pose:  
            break

# filter image is png or jpg, and cv2 read image and store in image_readed
# image_readed store ==> filename, reference_pose_xy | image_readed, width, height 
image_chosen_readed,image_chosen_point = [],[]
for i in image_passed:
    if filename[i].endswith(".jpg") or filename[i].endswith(".png"):   
        image_path = os.path.join(image_drawed, filename[i])          
        image = cv2.imread(image_path)
        image_chosen_readed.append(image)
        image_chosen_point.append(reference_pose[i])

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    image_campiang = 0
    frame_point = [(0,0)*33]
    while cap.isOpened() and image_campiang<=(round_pose-1):
        ret, frame = cap.read()
        if not ret:
            print("Failed to open webcam.")
            exit()

        #frame to show in window
        frame = nf.cam_dir(cam_type,win_cam,win_monitor,0,480,frame)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #process and draw on image
        results = pose.process(rgb_frame)
        if results.pose_landmarks: #draw pose landmark on your webcam
            pose_landmarks = [(lm.x, lm.y) for lm in results.pose_landmarks.landmark]

            #get ref and frame point
            ref_point = image_chosen_point[image_campiang]
            frame_prop = frame.shape[0:2]
            frame_point = pose_landmarks

            ref_point = nf.upZip(ref_point)
            frame_point = nf.upZip(frame_point)

            #use preason colliretion
            similarity = nf.preacole(ref_point,frame_point)
    
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                                mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                                )
            if similarity > 0.9:
                cv2.putText(
                frame,
                f"{image_campiang+1}. Similarity: {similarity:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
                            )
                continue
            else:
                cv2.putText(
                frame,
                f"{image_campiang+1}. Similarity: {similarity:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
                            )

        #insert ref image   
        ref = image_chosen_readed[image_campiang]
        ref = nf.resize_image_ref(win_monitor,ref)
        
        #show window 
        im_h = cv2.hconcat([ref,frame]) #link 2 image
        cv2.imshow("Pose Estimation", im_h)


        if cv2.waitKey(1) & 0xFF == ord("n"):
            image_campiang += 1
            continue
    
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break



    cap.release()
    cv2.destroyAllWindows()
        

