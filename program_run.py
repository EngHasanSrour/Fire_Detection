import cv2
from Alarm_Player import AlarmPlayer

def main():
    # Load the Haar Cascade model for fire detection
    cascade_path = 'fire_detection_cascade_model.xml'
    fire_cascade = cv2.CascadeClassifier(cascade_path)

    # Initialize webcam stream
    webcam = cv2.VideoCapture(0)

    # Initialize the sound player instance
    alarm_player = AlarmPlayer('fire_alarm.mp3')

    while True:
        ret, frame = webcam.read()
        if not ret:
            print("Error: Unable to capture frame from camera.")
            break

        # Convert frame to grayscale for cascade detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect fires in the frame
        fires = fire_cascade.detectMultiScale(gray_frame, scaleFactor = 1.2, minNeighbors = 5)

        # Handle detection results
        if len(fires) > 0:
            # Play alarm sound asynchronously without blocking video stream
            alarm_player.play()

            # Draw bounding boxes and labels around detected fires
            for x, y, w, h in fires:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), thickness = 2)
                cv2.putText(
                    frame,
                    "FIRE DETECTED",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    thickness = 2)
                print(f'Fire detected at coordinates: x={x}, y={y}, width={w}, height={h}')

        # Display the output frame
        cv2.imshow('Fire Detection System', frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources and cleanup
    webcam.release()
    cv2.destroyAllWindows()
    alarm_player.stop()


if __name__ == '__main__':
    main()