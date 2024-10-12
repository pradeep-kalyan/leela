import cv2
import argparse
import numpy as np

# Define constants
CONFIDENCE_THRESHOLD = 0.7
FACE_DETECTOR_SIZE = (300, 300)
AGE_DETECTOR_SIZE = (227, 227)
MODEL_MEAN_VALUES = (78.463377603, 87.7689143744, 114.895847746)

# Define age and gender lists
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(20-25)', '(30-40)', '(40-50)', '(60-80)', '(80-100)']
GENDER_LIST = ['Male', 'Female']

def load_face_detector(proto_path, model_path):
    """Load face detector model"""
    return cv2.dnn.readNet(model_path, proto_path)

def load_age_gender_models(age_proto_path, age_model_path, gender_proto_path, gender_model_path):
    """Load age and gender prediction models"""
    age_net = cv2.dnn.readNet(age_model_path, age_proto_path)
    gender_net = cv2.dnn.readNet(gender_model_path, gender_proto_path)
    return age_net, gender_net

def detect_faces(face_net, frame):
    """Detect faces in a frame"""
    blob = cv2.dnn.blobFromImage(frame, 1.0, FACE_DETECTOR_SIZE, MODEL_MEAN_VALUES, True, False)
    face_net.setInput(blob)
    detections = face_net.forward()
    face_boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > CONFIDENCE_THRESHOLD:
            x1 = int(detections[0, 0, i, 3] * frame.shape[1])
            y1 = int(detections[0, 0, i, 4] * frame.shape[0])
            x2 = int(detections[0, 0, i, 5] * frame.shape[1])
            y2 = int(detections[0, 0, i, 6] * frame.shape[0])
            face_boxes.append([x1, y1, x2, y2])
    return face_boxes

def predict_age_gender(age_net, gender_net, face):
    """Predict age and gender of a face"""
    blob = cv2.dnn.blobFromImage(face, 1.0, AGE_DETECTOR_SIZE, MODEL_MEAN_VALUES, swapRB=False)
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = AGE_LIST[age_preds[0].argmax()]
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = GENDER_LIST[gender_preds[0].argmax()]
    return age, gender

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image')
    args = parser.parse_args()

    face_detector_proto = "opencv_face_detector.pbtxt"
    face_detector_model = "opencv_face_detector_uint8.pb"
    age_deploy_proto = "age_deploy.prototxt"
    age_deploy_model = "age_net.caffemodel"
    gender_deploy_proto = "gender_deploy.prototxt"
    gender_deploy_model = "gender_net.caffemodel"

    face_net = load_face_detector(face_detector_proto, face_detector_model)
    age_net, gender_net = load_age_gender_models(age_deploy_proto, age_deploy_model, gender_deploy_proto, gender_deploy_model)

    video = cv2.VideoCapture(args.image if args.image else 0)
    padding = 20

    while cv2.waitKey(1) < 0:
        has_frame, frame = video.read()
        if not has_frame:
            cv2.waitKey()
            break

        face_boxes = detect_faces(face_net, frame)
        if not face_boxes:
            print("No face is detected")
            continue

        for face_box in face_boxes:
            face = frame[max(0, face_box[1] - padding):min(face_box[3] + padding, frame.shape[0] - 1),
                         max(0, face_box[0] - padding):min(face_box[2] + padding, frame.shape[1] - 1)]

            age, gender = predict_age_gender(age_net, gender_net, face)
            print(f'Gender: {gender}, Age: {age[1:-1]} years')

            cv2.putText(frame, f'{gender}, {age[1:-1]} years', (face_box[0], face_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Gender and Age", frame)

if __name__ == "__main__":
    main()