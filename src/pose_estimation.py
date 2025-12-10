# pose_estimation.py

import cv2
import mediapipe as mp
import numpy as np
import argparse


# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Function to Draw Skeleton, this function is responsible for the skelton draw.
def draw_landmarks(image, landmarks):
    mp_drawing.draw_landmarks(
        image,
        landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
    )

# Pose Estimation from Webcam
def pose_estimation(webcam_id=0, output_file=None):
    cap = cv2.VideoCapture(webcam_id)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    # Get video properties
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(5))

    if output_file:
        out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        print("[INFO] Starting Pose Estimation...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False

            # Make Pose Prediction
            results = pose.process(frame_rgb)

            # Draw Pose Landmarks
            frame.flags.writeable = True
            if results.pose_landmarks:
                draw_landmarks(frame, results.pose_landmarks)

            # Display Results
            cv2.imshow("Pose Estimation", frame)

            # Write to Output File
            if output_file:
                out.write(frame)

            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        if output_file:
            out.release()
        cv2.destroyAllWindows()
        print("[INFO] Pose estimation finished.")


# Save Pose Data to File 

def save_pose_data(results, output_file):
    with open(output_file, 'w') as file:
        if results.pose_landmarks:
            for i, landmark in enumerate(results.pose_landmarks.landmark):
                file.write(f"{i},{landmark.x},{landmark.y},{landmark.z}\n")
            print(f"[INFO] Pose data saved to {output_file}")


# Main Function
def main():
    parser = argparse.ArgumentParser(description="Real-time Pose Estimation using MediaPipe and OpenCV")
    parser.add_argument('--webcam', type=int, default=0, help="Webcam ID (default=0)")
    parser.add_argument('--output', type=str, help="Path to save output video (optional)")
    parser.add_argument('--pose_data', type=str, help="Path to save pose data (optional)")

    args = parser.parse_args()

    pose_estimation(args.webcam, args.output)

if __name__ == "__main__":
    main()

