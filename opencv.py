import cv2
import numpy as np
from keras.models import load_model
from skimage.transform import resize, pyramid_reduce

model = load_model('model.h5')

def get_square(image, square_size):
    
    height, width = image.shape    
    if(height > width):
      differ = height
    else:
      differ = width
    differ += 4


    mask = np.zeros((differ, differ), dtype = "uint8")

    x_pos = int((differ - width) / 2)
    y_pos = int((differ - height) / 2)

   
    mask[y_pos: y_pos + height, x_pos: x_pos + width] = image[0: height, 0: width]

 
    if differ / square_size > 1:
      mask = pyramid_reduce(mask, differ / square_size)
    else:
      mask = cv2.resize(mask, (square_size, square_size), interpolation = cv2.INTER_AREA)
    return mask


def keras_predict(model, image):
    data = np.asarray( image, dtype="int32" )
    
    pred_probab = model.predict(data)[0]
    pred_class = list(pred_probab).index(max(pred_probab))
    return max(pred_probab), pred_class

def keras_process_image(img):
    
    image_x = 28
    image_y = 28
    #img = cv2.resize(img, (28,28), interpolation = cv2.INTER_AREA)
    img = get_square(img, 28)
    img = np.reshape(img, (image_x, image_y))
    
    
    return img
 

def crop_image(image, x, y, width, height):
    return image[y:y + height, x:x + width]

while True:  
    cam_capture = cv2.VideoCapture(0)
    _, image_frame = cam_capture.read()  
    # Select ROI
    im2 = crop_image(image_frame, 300,300,300,300)
    image_grayscale = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    image_grayscale_blurred = cv2.GaussianBlur(image_grayscale, (15,15), 0)


    #resized_img = image_resize(image_grayscale_blurred, width = 28, height = 28, inter = cv2.INTER_AREA) 
    #resized_img = keras_process_image(image_grayscale_blurred)
    resized_img = cv2.resize(image_grayscale_blurred,(28,28))
    #ar = np.array(resized_img)
    ar = resized_img.reshape(1,784)

    pred_probab, pred_class = keras_predict(model, ar )
    print(pred_class, pred_probab)
     
    
 
    # Display cropped image

    cv2.imshow("Image2",im2)
    cv2.imshow("Image4",resized_img)
    cv2.imshow("Image3",image_grayscale_blurred)

    if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    

cam_capture.release()
cv2.destroyAllWindows()
