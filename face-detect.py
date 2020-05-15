import cv2
import dlib

cap = cv2.VideoCapture(0)

hog_face_detector = dlib.get_frontal_face_detector()
while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    faces_hog = hog_face_detector(frame)
    for face in faces_hog:
        x = face.left()
        y = face.top()
        w = face.right() - x
        h = face.bottom() - y
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
