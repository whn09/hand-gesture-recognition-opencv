import cv2
import numpy as np
from PIL import Image
from HandRecognition import hand_capture, hand_threshold, hand_contour_find, mark_hand_center, mark_fingers, \
    find_gesture


def balance_color(filename):
    im = Image.open(filename)
    tb = im.histogram()

    totalpixel = 0
    maptb = []
    count = len(tb)
    for i in range(count):
        totalpixel += tb[i]
        maptb.append(totalpixel)

    for i in range(count):
        maptb[i] = int(round((maptb[i] * (count - 1)) / totalpixel))

    def histogram(light):
        return maptb[light]

    out = im.point(histogram)
    #out.show()
    #return cv2.cvtColor(np.asarray(out), cv2.COLOR_RGB2BGR)
    #print(np.asarray(out))
    return np.asarray(out)


def hand_recognition_single(filename):
    #frame = cv2.imread(filename)
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame = balance_color(filename)

    frame_original = np.copy(frame)
    cv2.imwrite('1_frame_original.png', frame_original)

    ret, frame = cv2.threshold(frame, 225, 255, 0)  # default: 127, 255, 0
    for y in range(int(frame.shape[1]*0.3), int(frame.shape[1]*0.8)):  # TODO TRICKY!
        frame[frame.shape[0]-1][y] = 255
    cv2.imwrite('2_threshold.png', frame)

    #kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite('3_morphologyEx.png', frame)

    frame = cv2.medianBlur(frame, 3)
    cv2.imwrite('4_medianBlur.png', frame)

    frame, contours, hierarchy = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imwrite('5_contour.png', frame)

    found, hand_contour = hand_contour_find(contours)
    print('hand_contour:', len(hand_contour))
    frame = cv2.drawContours(frame_original, [hand_contour], -1, (0, 0, 255, 255), thickness=-1)
    cv2.imwrite('6_hand_contour.png', frame)

    alpha_shape = (frame.shape[0], frame.shape[1], 4)
    hand = np.zeros(alpha_shape, np.uint8)
    hand = cv2.drawContours(hand, [hand_contour], -1, (0, 0, 255, 255))
    epsilion = hand.shape[0] / 200
    approxe = cv2.approxPolyDP(hand_contour, epsilion, True)
    hand = cv2.polylines(hand, [approxe], True, (255, 0, 0, 255), 1)  # green
    cv2.imwrite('hand.png', hand)

    if found:
        hand_convex_hull = cv2.convexHull(hand_contour)
        frame, hand_center, hand_radius, hand_size_score = mark_hand_center(frame_original, hand_convex_hull)
        cv2.imwrite('7_mark_hand_center_simple.png', frame)
        if hand_size_score:
            frame, finger, palm = mark_fingers(frame, hand_convex_hull, hand_center, hand_radius)
            cv2.imwrite('8_mark_fingers.png', frame)
            frame, gesture_found = find_gesture(frame, finger, palm)
            print('getsture_found:', gesture_found)
            cv2.imwrite('9_find_gesture.png', frame)
        else:
            print('not hand_size_score:', hand_size_score)
    else:
        print('not found')


if __name__ == '__main__':
    #filename = 'wanghenan_all.png'
    #filename = 'wanghenan_all.png.out.png'
    filename = 'WechatIMG106.jpg.out.png'
    #filename = 'WechatIMG107.jpg.out.png'
    hand_recognition_single(filename)
