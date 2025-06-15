import cv2
import numpy as np
import winsound

def nothing(x):
    pass

cv2.namedWindow("Trackbars")

cv2.createTrackbar("L-H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L-S", "Trackbars", 100, 255, nothing)
cv2.createTrackbar("L-V", "Trackbars", 100, 255, nothing)
cv2.createTrackbar("U-H", "Trackbars", 10, 179, nothing)
cv2.createTrackbar("U-S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U-V", "Trackbars", 255, 255, nothing)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("L-H", "Trackbars")
    l_s = cv2.getTrackbarPos("L-S", "Trackbars")
    l_v = cv2.getTrackbarPos("L-V", "Trackbars")
    u_h = cv2.getTrackbarPos("U-H", "Trackbars")
    u_s = cv2.getTrackbarPos("U-S", "Trackbars")
    u_v = cv2.getTrackbarPos("U-V", "Trackbars")

    lower_bound = np.array([l_h, l_s, l_v])
    upper_bound = np.array([u_h, u_s, u_v])

    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Apply Gaussian blur to reduce noise
    mask = cv2.GaussianBlur(mask, (9, 9), 0)

    res = cv2.bitwise_and(frame, frame, mask=mask)

    # Count the number of non-zero (white) pixels in the mask
    color_pixels = cv2.countNonZero(mask)

    # Only beep if enough of the color is present
    if color_pixels > 10000:  # ← Increase this to reduce false positives
        winsound.Beep(1000, 200)

    cv2.imshow("frame", frame)
    cv2.imshow("res", res)

    if cv2.waitKey(1) == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
    