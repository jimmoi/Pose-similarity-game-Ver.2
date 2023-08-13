import os
import cv2
import mediapipe as mp
# import uuid
from config import image_scale as fac

#rePixelsize of the longest side equal to fac
def scale_imsize(height, width,fac):
    h = height
    w = width
    fac = int(fac)
    if h<=fac & w<=fac:
        h,w=h,w
    elif h>=w:
        h,w = fac,int(w*(fac/h))
    else:
        w,h = fac,int(h*(fac/w))
    return(w,h)
        

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

image_folder = "raw ref img\put image here" #replace your image folder


# Create a new folder for each run
output_folder_path = 'Drawed reference image deletable/'
os.makedirs(output_folder_path, exist_ok=True)

#delete all file in folder
for filename in os.listdir(image_folder):
    file_path = os.path.join(output_folder_path, filename)
    if os.path.isfile(file_path):
        os.remove(file_path)


#set variable
all_reference_landmarks = []
number = 0
new_file_name = "set_{number}"

# Process each image in the folder
for filename in os.listdir(image_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Adjust the file extensions as needed
        # this will return a tuple of root and extension
        split_tup = os.path.splitext(filename)
        file_extension = split_tup[1]

        image_path = os.path.join(image_folder, filename)
        image = cv2.imread(image_path)
        height, width, _ = image.shape
        reSized = scale_imsize(height, width, fac)
        if fac > 100:
            image = cv2.resize(image,reSized)

    
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_image)

            if results.pose_landmarks:
                pose_landmarks = [(lm.x, lm.y) for lm in results.pose_landmarks.landmark]
                all_reference_landmarks.append(pose_landmarks)

                pose_landmarks = [(int(x),int(y)) for (x,y) in pose_landmarks]

                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                                        mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                                        )

                output_path = os.path.join(output_folder_path, f"set_{number:02}{file_extension}")
                cv2.imwrite(output_path, image)

            else:
                print(f"No pose landmarks found in {filename}")
    number += 1


# Save all pose landmarks to a text file
reference_file_path = "all_reference_poses_point_deletable.py"
with open(reference_file_path, "w") as f:
    f.write("reference_pose = [")
    for landmarks in all_reference_landmarks:
        f.write("[\n")
        for landmark in landmarks:
            f.write(f"({landmark[0]},{landmark[1]}),\n")
        f.write("],\n")
    f.write("]")

