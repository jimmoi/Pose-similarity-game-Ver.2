import cv2
import numpy as np
from config import scale,cam_type,window_name

def rotateResize_H2V_frame(win_cam,win_monitor,image):
    win_h, win_w = win_monitor
    width, height = win_cam
    rotation_matrix = cv2.getRotationMatrix2D((width/2, width/2), 90, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (int(height),int(width)))
    rotated_image = cv2.resize(rotated_image,(int(win_w*0.5625), int(win_h*0.5625)))
    rotated_image = cv2.flip(rotated_image, 1)
    return rotated_image

def crop_image(win_cam,win_monitor,Ver,Hor,image):
    win_w,win_h = win_cam
    winmo_w,winmo_h = win_monitor
    center_fromVertical = int(winmo_h/2)
    center_fromHorizon = int(winmo_w/2)
    new_frame = cv2.resize(image,(winmo_w,winmo_h))
    new_frame = cv2.flip(new_frame, 1)
    new_frame = new_frame[0+Ver : winmo_h-Ver , 0+Hor : winmo_w-Hor]
    return new_frame

def cam_dir(cam_type,win_cam,win_monitor,Ver,Hor,image):
    if cam_type in ["Hor","Ver"]:
        if cam_type == "Hor":
            frame = crop_image(win_cam,win_monitor,Ver,Hor,image)
        else:
            frame = rotateResize_H2V_frame(win_cam,win_monitor,image)
        return frame
    else:print("Valid cam type")

def resize_image_ref(win_monitor,image):
    image_prop = image.shape[:2]
    image_ref = cv2.resize(image,(int(image_prop[1]*(win_monitor[1]/image_prop[0])),win_monitor[1]))
    return image_ref

def upZip(point):
    all_xy = list(zip(*point))
    all_point = all_xy[0]+all_xy[1]
    all_point = np.array(all_point)
    return all_point

def preacole_shape(shape1,shape2):
    def preacole_dot(x_dot,y_dot):
        x,y = x_dot,y_dot
        relation_above = np.sum((x-np.mean(x)) * (y-np.mean(y)))
        relation_below = np.sqrt(np.sum((x-np.mean(x))**2) * np.sum((y-np.mean(y))**2))  
        results = relation_above/relation_below
        return results
    x1,y1 = zip(*shape1)
    x2,y2 = zip(*shape2)
    x_compare = preacole_dot(x1,x2)
    y_compare = preacole_dot(y1,y2)
    result = (x_compare+y_compare)/2
    result = (result,x_compare,y_compare)
    return result


if __name__ == "__main__":

    import cv2
    import numpy as np
    from config import scale,cam_type,window_name


    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,2000);  cap.set(cv2.CAP_PROP_FRAME_WIDTH,2000)

    w_cam = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h_cam = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    win_cam = (int(w_cam),int(h_cam))
    win_monitor = (int(w_cam*scale),int(h_cam*scale))

    image_ref_path = "A:\mediapipe/ver.1\Drawed reference image deletable\set_06.jpg"
    image_ref = cv2.imread(image_ref_path)
    image_prop = image_ref.shape[:2]
    image_ref = cv2.resize(image_ref,(int(image_prop[1]*(win_monitor[1]/image_prop[0])),win_monitor[1]))


    while cap.isOpened():
        ret, frame = cap.read()
        h,w,_ = frame.shape
        if not ret:
            print("Failed to open webcam.")
            exit()


        test = cam_dir(cam_type,win_cam,win_monitor,0,480,frame)
        

        im_h = cv2.hconcat([image_ref,test])
        cv2.imshow(window_name, im_h)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Cam is stopped")
            break
    cap.release()
    cv2.destroyAllWindows()