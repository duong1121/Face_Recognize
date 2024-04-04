import datetime
from ultralytics import YOLO
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
#import helper.face_match as match
from datetime import datetime
import torch
import numpy as np
import csv
from tensorflow.keras.models import load_model
from keras_vggface import utils
import pickle
import keras.utils as image

stream_url = 'rtsp://admin:FADUEU@169.254.126.112:554/H.264'
Save_Path = 'C:/Users/ACER/Downloads/Newsest_Face_Rec/Save_img/'
RESET = 30
CONFIDENCE_THRESHOLD = 0.7
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)

max_track = 0
NAMES_val = [0] * 50
name = None
C_Frame = 0
Store_name = np.empty(50, dtype=object)
ID_Check = []
ID_Appears = []
Time_In = [""] * 50


cap = cv2.VideoCapture(0)

# Set tracker for tracking face
tracker = DeepSort( max_iou_distance=0.9, max_age=20, n_init=10 )

# Load the pre-trained model face recognize
model = load_model('model.h5')

# Load the YOLO face detect model
model1 = YOLO('yolov8n-face.pt')
model1.to('cuda')

# Load label-name
face_label_filename = 'face-labels.pickle'
with open(face_label_filename, "rb") as \
    f: class_dictionary = pickle.load(f)
class_list = [value for _, value in class_dictionary.items()]


while True:
    # Start time to compute the fps
    #start = datetime.datetime.now()

    ret, frame = cap.read()
    if not ret:
        break
    
    detections = model1(frame,classes=[0,1])[0]
    results = []
    

    for data in detections.boxes.data.tolist():
        confidence = data[4]


        # Get info if cof high than threshold setted
        if float(confidence) < CONFIDENCE_THRESHOLD:
            continue
        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        class_id = data[5]
        results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id])
        

    # TRACKING
    # Update the tracker with the new detections
    tracks = tracker.update_tracks(results, frame=frame)
    
    
    if C_Frame == RESET:
        for track in tracks:
            # If the track is not confirmed, ignore it
            if not track.is_confirmed():
                continue


            # Get the track id and the bounding box
            track_id = track.track_id
            ltrb = track.to_ltrb()
            xmin, ymin, xmax, ymax = int(ltrb[0]), int(
                ltrb[1]), int(ltrb[2]), int(ltrb[3])
            

            # Process image
            img2 = frame[ymin:ymax, xmin:xmax]
            if img2.size <=0:
                break
            resized_image = cv2.resize(img2, (224,224))
            x = image.img_to_array(resized_image)
            x = np.expand_dims(x, axis=0)
            x = utils.preprocess_input(x, version=1)

            # Make a predicton
            predicted_prob = model.predict(x)


            # Store result (name)
            NAMES_val[int(track.track_id)-1] = NAMES_val[int(track.track_id)-1] + predicted_prob
            Store_name[int(track.track_id)-1] = class_list[predicted_prob[0].argmax()]
            track_id= Store_name[int(track.track_id)-1]
            print(track_id)
            name = track_id
            

            # Draw bounding box and class
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
            cv2.rectangle(frame, (xmin, ymin - 20), (xmin + 20, ymin), GREEN, -1)
            cv2.putText(frame, str(track_id), (xmin + 5, ymin - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)
            
        # Set the frame predict value
        C_Frame = 0
    
    else:
        for track in tracks:
            # If the track is not confirmed, ignore it
            if not track.is_confirmed():
                continue


            ltrb = track.to_ltrb()
            xmin, ymin, xmax, ymax = int(ltrb[0]), int(
                ltrb[1]), int(ltrb[2]), int(ltrb[3])
                
            # If a face getout of the frame get information
            if int(track.track_id)-1 not in ID_Appears:
                ID_Appears.append(int(track.track_id)-1)

            
                # Get the current time
                if Time_In[int(track.track_id)-1] == "":
                    GetTime_In = datetime.now()
                    current_hour = GetTime_In.hour
                    current_minute = GetTime_In.minute
                    time_value_in = f"{current_hour:02}:{current_minute:02}"
                    Time_In[int(track.track_id)-1]=(time_value_in)

                    Save_Path_img = Save_Path + str(int(track.track_id)-1) + ".jpg"
                    save_img = frame[ymin:ymax, xmin:xmax]
                    cv2.imwrite(Save_Path_img,save_img)


            # Get the track id and the bounding box
            track_id = track.track_id
            
            
            if xmin <= 0 or ymin <= 0 or xmax <= 0 or ymax <=0:
                pass

            if int(track.track_id) > max_track:

                # Proccess image
                img2 = frame[ymin:ymax, xmin:xmax]

                if img2.size <=0:
                    break

                resized_image = cv2.resize(img2, (224,224))
                x = image.img_to_array(resized_image)
                x = np.expand_dims(x, axis=0)
                x = utils.preprocess_input(x, version=1)


                # Make a predict
                predicted_prob = model.predict(x)


                # Get predict of this ID preson
                NAMES_val[int(track.track_id)-1] = NAMES_val[int(track.track_id)-1] + predicted_prob
                

                # Save instant name
                Store_name[int(track.track_id)-1] = class_list[predicted_prob[0].argmax()]
                track_id= Store_name[int(track.track_id)-1]

                name = track_id
                max_track += 1

            else:
                track_id = Store_name[int(track.track_id)-1]
            

            # Show the bounding boxes and classes
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
            cv2.rectangle(frame, (xmin, ymin - 20), (xmin + 20, ymin), GREEN, -1)
            cv2.putText(frame, str(track_id), (xmin + 5, ymin - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)
            

        # Get information if face out of the frame
        if ID_Check != 0 and ID_Check != ID_Appears:
            for value in ID_Check:
                if value not in ID_Appears:

                    # Get time out
                    GetTime_Out = datetime.now()
                    current_hour = GetTime_Out.hour
                    current_minute = GetTime_Out.minute
                    time_value_out = f"{current_hour:02}:{current_minute:02}"
                    Load_img = f'=HYPERLINK("file:///{Save_Path}{value}.jpg", "Click to Open Image")'
                    # Get information data
                    data = []
                    data.append(value)
                    data.append(class_list[NAMES_val[value].argmax()])
                    data.append(Time_In[value])
                    data.append(time_value_out)
                    data.append(Load_img)
                    # Write information to csv file
                    with open("data.csv", "a", newline="") as f:
                        csv_writer = csv.writer(f)
                        csv_writer.writerow(data)
                    print('value',value )
                    print(class_list[NAMES_val[value].argmax()])


        ID_Check = ID_Appears
        ID_Appears = []
        
    C_Frame += 1
    

    #end = datetime.datetime.now()
    # Show the time it took to process 1 frame
    #print(f"Time to process 1 frame: {(end - start).total_seconds() * 1000:.0f} milliseconds")
    #fps = f"FPS: {1 / (end - start).total_seconds():.2f}"
    #cv2.putText(frame, fps, (50, 50),
    #            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)


    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) == ord("q"):
        #print(NAMES_val)
        #print(class_list[NAMES_val[1].argmax()])
        break
        
cap.release()
cv2.destroyAllWindows()

