# import cv2

# face_cap = cv2.CascadeClassifier("")
# video_cap = cv2.VideoCapture(0)pip install deepface OpenCV TensorFlow
import cv2

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
video_capture = cv2.VideoCapture(0)

def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.7, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return faces

while True:

    result, video_frame = video_capture.read()  # read frames from the video
    if result is False:
        break  # terminate the loop if the frame is not read successfully

    faces = detect_bounding_box(
        video_frame
    )  # apply the function we created to the video frame

    cv2.imshow(
        "My Face Detection Project", video_frame
    )  # display the processed frame in a window named "My Face Detection Project"

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# while True:
#     ret , video_data = video_cap.read()
#     col = cv2.cvtColor(video_data,cv2.COLOR_BGR2GRAY)
#     faces = face_cap.detectMultiScale(
#         col,
#         scaleFactor= 1.1,
#         minNeighbors=5,
#         minSize=(30,30),
#         flags=cv2.CASCADE_SCALE_IMAGE
#     )
#     for(x,y,w,h) in faces:
#         cv2.rectangle(video_data,(x,y),(x+w,y+h),(0,255,0),2)
#         cv2.imshow("video_live",video_data)

#         if cv2.waitKey(10) == ord("a"):
#             break
#         video_cap.release()
'''import cv2
video_cap = cv2.VideoCapture(0)
while True :
    ret , video_data = video_cap.read()
    print(video_data)
    cv2.imshow("cameras",video_data)
    if cv2.waitKey(10) == ord("a"):
        break
    
  
video_cap.release()
cv2.destroyAllWindows() '''