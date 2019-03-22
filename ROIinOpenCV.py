import cv2
import numpy as np
from keras.models import load_model
from skimage.transform import resize, pyramid_reduce
import PIL
from PIL import Image

model = load_model('CNNmodel.h5')


def prediction(pred):
    if pred == 0:
        print('A')
    elif pred == 1:
        print('B')
    
    elif pred == 2:
        print('C')
    
    elif pred == 3:
        print('D')
    
    elif pred == 14:
        print('O')
    
    elif pred == 8:
        print('I')
    
    elif pred == 20:
        print('U')
    
    elif pred == 21:
        print('V')
    
    elif pred == 22:
        print('W')
    
    elif pred == 24:
        print('Y')
    
    elif pred == 11:
        print('L')
        


def keras_predict(model, image):
    data = np.asarray( image, dtype="int32" )
    
    pred_probab = model.predict(data)[0]
    pred_class = list(pred_probab).index(max(pred_probab))
    return max(pred_probab), pred_class

def keras_process_image(img):
    
    image_x = 28
    image_y = 28
    img = cv2.resize(img, (1,28,28), interpolation = cv2.INTER_AREA)
    #img = get_square(img, 28)
    #img = np.reshape(img, (image_x, image_y))
    
    
    return img
 

def crop_image(image, x, y, width, height):
    return image[y:y + height, x:x + width]

def main():
    while True:  
        cam_capture = cv2.VideoCapture(0)
        _, image_frame = cam_capture.read()  
    # Select ROI
        im2 = crop_image(image_frame, 300,300,300,300)
        image_grayscale = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    
        image_grayscale_blurred = cv2.GaussianBlur(image_grayscale, (15,15), 0)
        im3 = cv2.resize(image_grayscale_blurred, (28,28), interpolation = cv2.INTER_AREA)

    
    
    #ar = np.array(resized_img)
    #ar = resized_img.reshape(1,784)
    
    
    
        im4 = np.resize(im3, (28, 28, 1))
        im5 = np.expand_dims(im4, axis=0)
    

        pred_probab, pred_class = keras_predict(model, im5)
        #print(pred_class, pred_probab)
        prediction(pred_class)
    
 
    # Display cropped image

        cv2.imshow("Image2",im2)
    #cv2.imshow("Image4",resized_img)
        cv2.imshow("Image3",image_grayscale_blurred)

        if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

keras_predict(model, np.zeros((1, 28, 28, 1), dtype=np.uint8))
if __name__ == '__main__':
    main()

cam_capture.release()
cv2.destroyAllWindows()
