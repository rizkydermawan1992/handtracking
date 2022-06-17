import cv2
import mediapipe as mp
import numpy as np
import math
import pyfirmata

#input Field
indexCam = int(input("Masukkan Indeks Kamera: "))
port = input("Masukkan port COM Arduino: ")

#configuration opencv
ws, hs = 1280, 720
cap = cv2.VideoCapture(indexCam)
cap.set(3, ws)
cap.set(4, hs)
min_rat, max_rat = 20, 220
min_per, max_per = 0, 100
min_out, max_out = 0, 255


#configuration mediapipe
mpHand = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils
hands = mpHand.Hands( min_detection_confidence=0.8)

#configuration pyfirmata
board = pyfirmata.Arduino(port)
pin_pwm_1 = board.get_pin('d:6:p')  # pin 6 Arduino Motor DC
pin_pwm_2 = board.get_pin('d:9:p')  # pin 9 Arduino LED
pins  = [pin_pwm_1, pin_pwm_2]

def posFinger(finger_x, finger_y):
    pos = tuple(np.multiply(np.array((finger_x, finger_y)), [ws, hs]).astype(int))
    return pos

def calculateDistance(pos_1, pos_2):
    length = int(math.hypot(pos_1[0] - pos_2[0], pos_1[1] - pos_2[1]))
    return length

def markFinger(label, pos_1, pos_2, idx):
    if len(label) == 2:
       label = label[idx]

    #draw line
    cv2.line(img,pos_1, pos_2, (0, 0, 0), 4)
    cv2.circle(img,pos_1, 15, (0, 0, 0), cv2.FILLED)
    cv2.circle(img, pos_2, 15, (0, 0, 0), cv2.FILLED)

    #calculate distance
    length_ver = calculateDistance(pos_indextip, pos_thumbtip)
    length_hor = calculateDistance(pos_indexmcp, pos_pinkymcp)
    length_rat = int((length_ver / length_hor) * 100)
    length_per = int(np.interp(length_rat, [min_rat, max_rat], [min_per, max_per]))
    out_value = int(np.interp(length_rat, [min_rat, max_rat], [min_out, max_out]))

    # draw bbox hand
    lmList, xList, yList = [], [], []
    for lm in multiHandDetection[id].landmark:
        h, w, c = img.shape
        lm_x, lm_y = int(lm.x * w), int(lm.y * h)
        xList.append(lm_x)
        yList.append(lm_y)
        lmList.append([lm_x, lm_y])
        x_min, y_min = min(xList), min(yList)
        x_max, y_max = max(xList), max(yList)
        w_box, h_box = x_max - x_min, y_max - y_min

    cv2.rectangle(img, (x_min - 20, y_min - 20), (x_min + w_box + 20, y_min + h_box + 20),
                  (0, 0, 0), 4)
    cv2.rectangle(img, (x_min - 22, y_min - 20), (x_min + w_box + 22, y_min - 60),
                  (0, 0, 0), cv2.FILLED)
    cv2.putText(img, f'{label}', (x_min - 10, y_min - 27),
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 3)
    cv2.putText(img, f'{out_value}', (x_max - 45, y_min - 27),
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 3)

    #draw value bar
    length_bar = int(np.interp(length_per, [min_per, max_per], [x_min - 20, x_min + 300]))
    cv2.rectangle(img, (x_min - 20, y_min - 110), (length_bar, y_min-80), (0, 255, 255), cv2.FILLED)
    cv2.rectangle(img, (x_min - 20, y_min - 110), (x_min + 300, y_min - 80), (0, 0, 0), 4)
    cv2.putText(img, f'{length_per}%', (x_min + 310, y_min - 85),
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 3)

    print(f'Hand: {label}  Out Value: {out_value}')
    return out_value

while cap.isOpened():
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    multiHandDetection = results.multi_hand_landmarks
    # print(multiHandDetection)
    handType = results.multi_handedness
    # print(handType)

    if multiHandDetection:
        #Hand Visualization
        for id, lm in enumerate(multiHandDetection):
            mpDraw.draw_landmarks(img, lm, mpHand.HAND_CONNECTIONS,
                                  mpDraw.DrawingSpec(color=(0, 255, 255), thickness=4, circle_radius=7),
                                  mpDraw.DrawingSpec(color=(0, 0, 0), thickness=4))

            for idx, classification in enumerate(handType):
                pos_indextip = posFinger(lm.landmark[mpHand.HandLandmark.INDEX_FINGER_TIP].x,
                               lm.landmark[mpHand.HandLandmark.INDEX_FINGER_TIP].y)
                pos_thumbtip = posFinger(lm.landmark[mpHand.HandLandmark.THUMB_TIP].x,
                               lm.landmark[mpHand.HandLandmark.THUMB_TIP].y)
                pos_indexmcp = posFinger(lm.landmark[mpHand.HandLandmark.INDEX_FINGER_MCP].x,
                                lm.landmark[mpHand.HandLandmark.INDEX_FINGER_MCP].y)
                pos_pinkymcp = posFinger(lm.landmark[mpHand.HandLandmark.PINKY_MCP].x,
                               lm.landmark[mpHand.HandLandmark.PINKY_MCP].y)

                if len(multiHandDetection) == 2:
                    if classification.classification[0].index == id:
                        label = ["Left", "Right"]
                        out_value = markFinger(label, pos_indextip, pos_thumbtip, idx)
                else:
                    label = classification.classification[0].label
                    out_value = markFinger(label, pos_indextip, pos_thumbtip, idx)

            out_value = out_value/max_out
            pins[id].write(out_value)

    else:
        print("No Hand")


    cv2.imshow("Image", img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
