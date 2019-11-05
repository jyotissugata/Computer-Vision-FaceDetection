import numpy as np
import os
import math
import cv2

face_detect_object = cv2.face.LBPHFaceRecognizer_create()
person_names = os.listdir("dataset/train")

def get_path_list(root_path):
    person_names = os.listdir(root_path)
    return person_names
    
    '''
        To get a list of path directories from root path

        Parameters
        ----------
        root_path : str
            Location of root directory
        
        Returns
        -------
        list
            List containing the names of the sub-directories in the
            root directory
    '''

def get_class_names(root_path, train_names):
    path_list = os.listdir(root_path)
    image_path = []
    id_list = []

    for i, name in enumerate(path_list):

        person_images = os.listdir(root_path + '/' + name)

        for image in person_images:

            path = root_path + '/' + name + '/' + image
            image_path.append(path)

            id_list.append(i)
    
    return image_path, id_list
    '''
        To get a list of train images path and a list of image classes id

        Parameters
        ----------
        root_path : str
            Location of images root directory
        train_names : list
            List containing the names of the train sub-directories
        
        Returns
        -------
        list
            List containing all image paths in the train directories
        list
            List containing all image classes id
    '''

def get_train_images_data(image_path_list):

    train_image_list = []
    for path in image_path_list:
        image = cv2.imread(path,0)
        train_image_list.append(image)

    return train_image_list
    '''
        To load a list of train images from given path list

        Parameters
        ----------
        image_path_list : list
            List containing all image paths in the train directories
        
        Returns
        -------
        list
            List containing all loaded train images
    '''

def detect_faces_and_filter(image_list, image_classes_list = None):
    
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray_images = []
    rect_images = []
    faces = []
    images_id = []

    if (image_classes_list == None):
        root = "dataset/test"
        person_image = os.listdir(root)
        for image in person_image:

            gray = cv2.imread(root + '/' + image,0)
            gray_images.append(gray)
            detected_faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 5)
            
            if len(detected_faces) < 1:
                continue
            
            for face in detected_faces:
                x, y, w, h = face
                face_rect = gray[y:y+h , x:x+w]
                #crop face from gray

                rect_images.append(face_rect)
                faces.append(face)
                
            
        return rect_images, faces, images_id

    else :
        person_names = os.listdir("dataset/train")
        for i, name in enumerate(person_names):

            person_image = os.listdir("dataset/train" + '/' + name)

            for image in person_image:

                gray = cv2.imread("dataset/train" + '/' + name + '/' + image,0)
                gray_images.append(gray)
                detected_faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 5)
                
                if len(detected_faces) < 1:
                    continue
                
                for face in detected_faces:
                    x, y, w, h = face
                    face_rect = gray[y:y+h , x:x+w]
                    #crop face from gray

                    rect_images.append(face_rect)
                    faces.append(face)
                    images_id.append(image_classes_list[i])
        
        return rect_images, faces, images_id
    '''
        To detect a face from given image list and filter it if the face on
        the given image is more or less than one

        Parameters
        ----------
        image_list : list
            List containing all loaded images
        image_classes_list : list, optional
            List containing all image classes id
        
        Returns
        -------
        list
            List containing all filtered and cropped face images in grayscale
        list
            List containing all filtered faces location saved in rectangle
        list
            List containing all filtered image classes id
    '''

def train(train_face_grays, image_classes_list):

    
    face_recognizer = face_detect_object.train(train_face_grays, np.array (image_classes_list))
    return face_recognizer
    '''
        To create and train classifier object

        Parameters
        ----------
        train_face_grays : list
            List containing all filtered and cropped face images in grayscale
        image_classes_list : list
            List containing all filtered image classes id
        
        Returns
        -------
        object
            Classifier object after being trained with cropped face images
    '''

def get_test_images_data(test_root_path, image_path_list):
    test_image_list = []
    for path in image_path_list:
        image_path = test_root_path + '/' + path
        image = cv2.imread(image_path)
        test_image_list.append(image)

    return test_image_list
    '''
        To load a list of test images from given path list

        Parameters
        ----------
        test_root_path : str
            Location of images root directory
        image_path_list : list
            List containing all image paths in the test directories
        
        Returns
        -------
        list
            List containing all loaded test images
    '''

results = []

def predict(classifier, test_faces_gray):

    predictions = []
    #root = "dataset/test"
    #test_image = os.listdir(root)
    for face in test_faces_gray:
        #x, y, w, h = face
        #face_rect = face[y:y+h , x:x+w]

        result, confidence = face_detect_object.predict(face)
        predictions.append(result)

    return predictions

    '''
        To predict the test image with classifier

        Parameters
        ----------
        classifier : object
            Classifier object after being trained with cropped face images
        train_face_grays : list
            List containing all filtered and cropped face images in grayscale

        Returns
        -------
        list
            List containing all prediction results from given test faces
    '''

def draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names):

    predicted_test_image_list = []
    for i, image in enumerate(test_image_list):
        #display result
        x,y,w,h = test_faces_rects[i]
        text = train_names[predict_results[i]]

        cv2.rectangle(image, (x,y), (x+w, y+h), (255,255,0), 1)

        predicted_test_image_list.append(cv2.putText(image, text, (x, y-10), 0, 1, (255,255,0)))

    return predicted_test_image_list
        
    '''
        To draw prediction results on the given test images

        Parameters
        ----------
        predict_results : list
            List containing all prediction results from given test faces
        test_image_list : list
            List containing all loaded test images
        test_faces_rects : list
            List containing all filtered faces location saved in rectangle
        train_names : list
            List containing the names of the train sub-directories

        Returns
        -------
        list
            List containing all test images after being drawn with
            prediction result
    '''

def combine_results(predicted_test_image_list):
    images_combine = np.hstack((predicted_test_image_list[0], predicted_test_image_list[1]))
    for index in range(3, len(predicted_test_image_list)):
        if(index <= len(predicted_test_image_list) -1):
            images_combine = np.hstack((images_combine, predicted_test_image_list[index]))
    return images_combine
    '''
        To combine all predicted test image result into one image

        Parameters
        ----------
        predicted_test_image_list : list
            List containing all test images after being drawn with
            prediction result

        Returns
        -------
        ndarray
            Array containing image data after being combined
    '''

def show_result(image):
    cv2.imshow("The Result", image)
    cv2.waitKey(0)
    '''
        To show the given image

        Parameters
        ----------
        image : ndarray
            Array containing image data
    '''

'''
You may modify the code below if it's marked between

-------------------
Modifiable
-------------------

and

-------------------
End of modifiable
-------------------
'''
if __name__ == "__main__":
    '''
        Please modify train_root_path value according to the location of
        your data train root directory

        -------------------
        Modifiable
        -------------------
    '''
    train_root_path = "dataset/train"
    '''
        -------------------
        End of modifiable
        -------------------
    '''
    
    train_names = get_path_list(train_root_path)
    image_path_list, image_classes_list = get_class_names(train_root_path, train_names)
    train_image_list = get_train_images_data(image_path_list)
    train_face_grays, _, filtered_classes_list = detect_faces_and_filter(train_image_list, image_classes_list)
    classifier = train(train_face_grays, filtered_classes_list)

    '''
        Please modify test_image_path value according to the location of
        your data test root directory

        -------------------
        Modifiable
        -------------------
    '''
    test_root_path = "dataset/test"
    '''
        -------------------
        End of modifiable
        -------------------
    '''
    
    test_names = get_path_list(test_root_path)
    test_image_list = get_test_images_data(test_root_path, test_names)
    test_faces_gray, test_faces_rects, _ = detect_faces_and_filter(test_image_list, None)
    predict_results = predict(classifier, test_faces_gray)
    predicted_test_image_list = draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names)
    final_image_result = combine_results(predicted_test_image_list)
    show_result(final_image_result)