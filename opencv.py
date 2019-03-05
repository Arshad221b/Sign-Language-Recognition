import cv2
from matplotlib import pyplot as plt
import numpy as np 
from keras.models import load_model

model = load_model('model.h5')



def keras_predict(model, image):
    data = np.asarray( image, dtype="int32" )
    
    pred_probab = model.predict(data)[0]
    pred_class = list(pred_probab).index(max(pred_probab))
    return max(pred_probab), pred_class



def keras_process_image(img):
    image_x = 28
    image_y = 28
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (-1, image_x, image_y, 1))
    return img







def sketch_transform(image):
    image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_grayscale_blurred = cv2.GaussianBlur(image_grayscale, (15,15), 0)
    
    
    return image_grayscale_blurred




cam_capture = cv2.VideoCapture(0)

upper_left = (300, 300)
bottom_right = (1084, 1084)



while True:
    cam_capture = cv2.VideoCapture(0)

    upper_left = (300, 300)
    bottom_right = (1084, 1084)

    _, image_frame = cam_capture.read()
    
    #Rectangle marker
    r = cv2.rectangle(image_frame, upper_left, bottom_right, (100, 50, 200), 5)
    rect_img = image_frame[upper_left[1] : bottom_right[1], upper_left[0] : bottom_right[0]]
    
    sketcher_rect = rect_img
    sketcher_rect = sketch_transform(sketcher_rect)
    
    
    
    cv2.resize(sketcher_rect, (28,28), interpolation = cv2.INTER_AREA)
    pred_probab, pred_class = keras_predict(model, sketcher_rect)
    print(pred_class, pred_probab)
    cv2.imshow('image_frame',sketcher_rect)

    

    if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        
cam_capture.release()
cv2.destroyAllWindows()