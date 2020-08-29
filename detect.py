import os
import cv2
import numpy as np
from keras.models import load_model

# Define paths
base_dir = os.path.dirname(__file__)
prototxt_path = os.path.join(base_dir + '/model_data/deploy.prototxt') #Weight and model path
caffemodel_path = os.path.join(base_dir + '/model_data/weights.caffemodel') 
font = cv2.FONT_HERSHEY_SIMPLEX

# Read the model
model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path) #load model


def main ():
    video_capture = cv2.VideoCapture(0) 
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640,480))
    model_mask = load_model('detect.h5') #load mask detection model
    while True:
        _, image = video_capture.read() 

        (h, w) = image.shape[:2] #get height and width of the model
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

        model.setInput(blob)
        detections = model.forward()
        
        for i in range(0, detections.shape[2]):
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            confidence = detections[0, 0, i, 2]
            if (confidence > 0.5):
                crop_img = image[startY:endY, startX:endX]
                crop_img = cv2.resize(crop_img, (100,100))
                crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
                prediction = model_mask.predict([crop_img.reshape(-1,100,100,1)])[0]
                
                if prediction[0] >= 0.5: #mask is on
                    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    cv2.putText(image, "MASK", (startX, startY), font, 0.5, (0, 255, 0), 2)
                #elif prediction[1] > 0.75: #mask is off
                else:
                    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
                    cv2.putText(image, "NO MASK", (startX, startY), font, 0.5, (0, 0, 255), 2)
                #else:
                    #cv2.rectangle(image, (startX, startY), (endX, endY), (255, 255, 255), 2)
                    #cv2.putText(image, "Not sure", (startX, startY), font, 0.5, (255, 255, 255), 2)


        cv2.imshow('Video', image)
        out.write(image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break 
    video_capture.release()
    cv2.destroyAllWindows()

main()