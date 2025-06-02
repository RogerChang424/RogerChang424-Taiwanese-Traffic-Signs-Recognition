import cv2
import numpy as np
from scipy import special
import time

classes = ['Max. 40', 'Max. 50', 'Max. 60',  'Max. 70',
           'Max. 80', 'Max. 90', 'Max. 100', 'Max. 110',
           'road separation', 'right confluence', 'left confluence', 'car/motorcycle only', 'car only', '2-step left turn',
           'aware of traffic light',
           'Min. 60', 'aware of speed traps', 'no right turns', 'no left/right turns', 'no left turns',
           'no U-turns', 'no parking', 'no entry', 'no temp. parking',
           'slow', 'turn lights on', 'drive on the right', 'none']

# class name index to class index of model
lookup = np.array([0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 24, 25, 26, 27, 3, 4, 5, 6, 7, 8, 9])


INPUT_HEIGHT = 360
INPUT_WIDTH  = (INPUT_HEIGHT * 4) // 3
criterion    = 95

cam_index    = 0
show_img = True
for i in range (28):
    print(classes[i])


    

# initializing cam
cap = cv2.VideoCapture(cam_index) # video capture source camera (Here webcam of laptop) 
cap.set(3, INPUT_WIDTH) #width=640
cap.set(4, INPUT_HEIGHT) #height=480

net = cv2.dnn.readNetFromONNX("model.onnx")



# first image
ret,frame = cap.read()
t = time.localtime()
current_time = time.strftime("%H:%M:%S", t)

while(ret):

    frame = cv2.imread("./test 1.png")
    frame = cv2.resize(frame, [INPUT_WIDTH, INPUT_HEIGHT])

    # 40 * 40 kernel, 20 px per step
    for h_steps in range((INPUT_HEIGHT-40)//20 + 1):
        for l_steps in range((INPUT_WIDTH-40)//20 + 1):
            up_left    = [l_steps * 20, h_steps * 20]
            down_right = [l_steps * 20 + 40, h_steps * 20 + 40]
            kernel = frame[up_left[1]: down_right[1], up_left[0]: down_right[0], :]
            kernel = cv2.cvtColor(kernel, cv2.COLOR_BGR2RGB).astype(np.float32)
            kernel /= 255.0
            kernel *= 2.0
            kernel -= 1.0


            blob = cv2.dnn.blobFromImage(kernel)


            net.setInput(blob)
            output = net.forward()
            
            net.setInput(blob)
            output = special.softmax(net.forward()[0])

            pred = np.argmax(output).astype(int)
            conf = output[pred]
            result = lookup[pred]
            #print(output)
            #if(True):
            if(conf > criterion/100.0 and result < 27):
                org = [up_left[0], up_left[1]]
                if(org[1] >= 10):
                    org[1] -= 10
                else:
                    org[1] = down_right[1] + 20
                print("size: 40")
                print("class: " + classes[result])
                print("conf: "  + str(output[pred]))
                frame = cv2.putText(frame, classes[result], org = org, fontFace = cv2.FONT_HERSHEY_DUPLEX,
                                    fontScale = 0.5, color = [0, 200, 200], thickness = 1)
                frame = cv2.rectangle(frame, up_left, down_right, color=[0, 150, 150], thickness=2)

    # 80 * 80 kernel, 40 px per step
    for h_steps in range((INPUT_HEIGHT-80)//40 + 1):
        for l_steps in range((INPUT_WIDTH-80)//40 + 1):
            up_left    = [l_steps * 40, h_steps * 40]
            down_right = [l_steps * 40 + 80, h_steps * 40 + 80]

            kernel = frame[up_left[1]: down_right[1], up_left[0]: down_right[0], :]
            kernel = cv2.cvtColor(kernel, cv2.COLOR_BGR2RGB).astype(np.float32)
            kernel = cv2.resize(kernel, [40, 40])

            kernel /= 255.0
            kernel *= 2.0
            kernel -= 1.0

            blob = cv2.dnn.blobFromImage(kernel)

            net.setInput(blob)
            output = special.softmax(net.forward()[0])

            pred = np.argmax(output).astype(int)
            conf = output[pred]
            result = lookup[pred]

            if(conf > criterion/100.0 and result < 27):
                org = [up_left[0], up_left[1]]
                if(org[1] >= 10):
                    org[1] -= 10
                else:
                    org[1] = down_right[1] + 20
                print("size: 80")
                print("class: " + classes[result])
                print("conf: "  + str(output[pred]))
                frame = cv2.putText(frame, classes[result], org = org, fontFace = cv2.FONT_HERSHEY_DUPLEX,
                                    fontScale = 0.5, color = [0, 200, 200], thickness = 1)
                frame = cv2.rectangle(frame, up_left, down_right, color=[0, 100, 100], thickness=2)
    

    if(show_img):
        frame = cv2.resize(frame, [1280, 960])
        cv2.imshow('', frame)
        cv2.waitKey(5000)

    #cv2.imshow('captured',norm) #display the captured image
    ret, frame = cap.read()
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    
    key = cv2.waitKey(100)
    if key == 27:
        break;

cv2.destroyAllWindows()
cap.release()
    