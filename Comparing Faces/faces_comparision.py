import numpy as np
import cv2
import face_recognition as fr
from glob import glob
import random

FACES_DIR = "faces_to_train"
FACES_TO_TEST_DIR = "faces_to_test"

faces = glob(FACES_DIR + '/*.*')
faces_to_test = glob(FACES_TO_TEST_DIR + '/*.*')

def recognize():
    # Generate random index for faces and faces_to_test
    n_random_face = random.randint(0,len(faces)-1)
    n_random_face_to_test = random.randint(0,len(faces_to_test)-1)

    # Load random image file
    img = fr.load_image_file(faces[n_random_face], mode='RGB')
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_test = fr.load_image_file(faces_to_test[n_random_face_to_test], mode='RGB')
    img_test = cv2.cvtColor(img_test,cv2.COLOR_BGR2RGB)

    # Detect face locations, face_locations - Returns an array of bounding boxes of human faces in a image
    img_face_loc = fr.face_locations(img)
    img_test_face_loc = fr.face_locations(img_test)

    # Draw Face locations on image
    for index, loc in enumerate(img_face_loc):
        cv2.rectangle(img,(loc[3],loc[0]),(loc[1],loc[2]),(0,0,255),2) # top-right, bottom-left
        cv2.putText(img,f'Face-{index+1}',(loc[3],loc[0]),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)
    for index, loc in enumerate(img_test_face_loc):
        cv2.rectangle(img_test,(loc[3],loc[0]),(loc[1],loc[2]),(0,0,255),2) # top-right, bottomleft
        cv2.putText(img_test,f'Face-{index+1}',(loc[3],loc[0]),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)


    # Encode the face features into 128-dimention value, (2nd parameter is optional)
    img_face_encodings = fr.face_encodings(img, img_face_loc)
    img_test_face_encodings = fr.face_encodings(img_test, img_test_face_loc)

    # Print currennt files
    print(f"Files - {faces[n_random_face].split('/')[-1]} AND {faces_to_test[n_random_face_to_test].split('/')[-1]}\n")

    # For every face in test image, check for match with score
    for index, encoding in enumerate(img_test_face_encodings):
        is_face_same = fr.compare_faces(img_face_encodings, encoding)
        score = fr.face_distance(img_face_encodings, encoding)

        try:
            if is_face_same[0]:
                print(f"Face {index+1}, matched with score {round(score[0], 3)}")
            else:
                print(f"Face {index+1}, did not match with score {round(score[0], 3)}")
        except Exception as e:
            print(f"Face {index+1} was not recognized, try again")

    print('---------------------------------------------------------\n')

    frame_size = 600
    result = cv2.hconcat((cv2.resize(img, (frame_size,frame_size)), cv2.resize(img_test, (frame_size,frame_size))))

    cv2.imshow('Result : Train and Test', result)
    cv2.waitKey(1000)

recognize()
is_continue = 'y'
while(is_continue == 'y'):
    is_continue = input('Do you want to continue? [y/n] : ')
    if(is_continue == 'y'):
        recognize()
    elif(is_continue == 'n'):
        print("Seems like you've seen enough")
    else:
        print('Oops! Provide a valid input')
        is_continue = 'y'
