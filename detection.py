import os
import matplotlib.pyplot as plt
import xmltodict
import random
from os import listdir
from os.path import isfile, join
import torchvision
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
import cvlib as cv
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import ImageFont, ImageDraw, Image
from training import load_checkpoint

if __name__ == '__main__':
    models_dir = "models/"
    filepath = models_dir + "15" + ".pth"
    loaded_model = load_checkpoint(filepath)
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # open webcam
    webcam = cv2.VideoCapture(0)
    font_scale = 1
    thickness = 2
    red = (0, 0, 255)
    green = (0, 255, 0)
    blue = (255, 0, 0)
    yellow = (0, 204, 255)
    white = (255, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX

    if not webcam.isOpened():
        print("Could not open webcam")
        exit()

    # loop through frames
    while webcam.isOpened():

        # read frame from webcam
        status, frame = webcam.read()

        if not status:
            print("Could not read frame")
            exit()

        # apply face detection
        face, confidence = cv.detect_face(frame)

        # loop through detected faces
        for idx, f in enumerate(face):

            (startX, startY) = f[0] - 10, f[1]
            (endX, endY) = f[2] + 10, f[3] + 20

            if 0 <= startX <= frame.shape[1] and 0 <= endX <= frame.shape[1] and 0 <= startY <= frame.shape[
                0] and 0 <= endY <= frame.shape[0]:

                face_region = frame[startY:endY, startX:endX]

                face_region1 = cv2.resize(face_region, (224, 224), interpolation=cv2.INTER_AREA)

                pil_image = Image.fromarray(face_region1, mode="RGB")
                pil_image = train_transforms(pil_image)
                image = pil_image.unsqueeze(0)

                result = loaded_model(image)
                _, maximum = torch.max(result.data, 1)
                prediction = maximum.item()

                if prediction == 0:
                    Y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.putText(frame, "Chin mask! please wear it correct way", (startX, Y), font, font_scale, green,
                                thickness)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), green, 2)
                elif prediction == 1:
                    Y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.putText(frame, "Nose mask! put the mask up to your nose ", (startX, Y), font, font_scale, blue,
                                thickness)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), blue, 2)
                elif prediction == 2:
                    Y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.putText(frame, "Valve mask, please wear the proper mask", (startX, Y), font, font_scale, white,
                                thickness)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), white, 2)
                elif prediction == 3:
                    Y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.putText(frame, "Good masker!", (startX, Y), font, font_scale, yellow, thickness)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), yellow, 2)
                elif prediction == 4:
                    Y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.putText(frame, "No mask...Please wear a mask!", (startX, Y), font, font_scale, red, thickness)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), red, 2)

        # display output
        cv2.imshow("mask nomask classify", frame)

        # press "Q" to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # release resources
    webcam.release()
    cv2.destroyAllWindows()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
