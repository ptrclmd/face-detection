import face_recognition
import cv2
import numpy as np
import os


def lock_laptop():
    return (os.system("shutdown.exe /l"))


def main():

    # Get webcam #0 (default one)
    video_capture = cv2.VideoCapture(0)

    # Load a picture and learn how to recognize it.
    pic_to_compare = face_recognition.load_image_file(
        "C:\\Users\\ptrca\\Documents\\face_detection\\pic.jpg")  # path to the picture

    pic_to_compare_encoding = face_recognition.face_encodings(pic_to_compare)[
        0]

    # Create arrays of your face and your name
    my_face_encoding = [
        pic_to_compare_encoding,
    ]
    my_name = [
        "Patricia",
    ]

    face_locations = []
    face_encodings = []
    face_names = []
    process_frame = True
    flag_to_lock = ""

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Only process every other frame of video to save time
        if process_frame:
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (BGR is what OpenCV uses) to RGB color (which is used by the library face_recognition)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(
                rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(
                    my_face_encoding, face_encoding)
                name = "Unknown"

                # Use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(
                    my_face_encoding, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    # in this case there's only one name in the array which is mine but you could adapt the script to know more faces
                    name = my_name[best_match_index]

                face_names.append(name)

            if "Unknown" in face_names:
                video_capture.release()
                cv2.destroyAllWindows()
                return lock_laptop()
            elif "Patricia" in face_names:
                return

        process_frame = not process_frame


if __name__ == "__main__":
    main()
