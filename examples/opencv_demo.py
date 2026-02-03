"""
Demo: 3D Object Tracking với OpenCV

Chạy: python examples/opencv_demo.py

This demo shows how to use scanlt's 3D tracking capabilities with OpenCV.
Press 'q' to quit, 'r' to reset tracking.
"""

import sys

try:
    import cv2
except ImportError:
    print("Error: opencv-python is required to run this demo.")
    print("Install with: pip install opencv-python")
    sys.exit(1)

try:
    import scanlt
except ImportError:
    print("Error: scanlt3d is not installed.")
    print("Install with: pip install scanlt3d")
    sys.exit(1)


def main():
    """Run the 3D tracking demo."""
    print("=" * 60)
    print("scanlt3d - 3D Object Tracking Demo")
    print("=" * 60)
    print()
    print("Controls:")
    print("  - Press 'q' to quit")
    print("  - Press 'r' to reset tracking")
    print()
    
    # Check backend
    backend_info = scanlt.choose_backend()
    print(f"Backend: {backend_info.name} ({backend_info.reason})")
    print()
    
    # Create tracker with fast profile
    try:
        tracker = scanlt.Tracker3D(profile="fast")
        print("Tracker initialized successfully!")
        print()
    except Exception as e:
        print(f"Error initializing tracker: {e}")
        print("Continuing with basic mode...")
        tracker = scanlt.Tracker3D(profile="fast")
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_count = 0
    
    print("Starting tracking... (this may take a moment to load models)")
    print()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from webcam")
            break
        
        # Process frame with tracker
        try:
            annotated_frame, tracks = tracker.process(frame, draw_annotations=True)
            
            # Display tracking info in terminal (every 30 frames to avoid spam)
            if frame_count % 30 == 0 and tracks:
                print(f"\nFrame {frame_count}:")
                for t in tracks:
                    print(f"  [{t.id}] {t.class_name}: "
                          f"pos=({t.x:.2f}, {t.y:.2f}, {t.z:.2f}m), "
                          f"conf={t.confidence:.2f}, "
                          f"frames={t.frames_tracked}")
            
            # Add frame counter to display
            cv2.putText(
                annotated_frame,
                f"Frame: {frame_count} | Tracks: {len(tracks)}",
                (10, annotated_frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            
            # Show the frame
            cv2.imshow("scanlt3d - 3D Tracking", annotated_frame)
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            cv2.imshow("scanlt3d - 3D Tracking", frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nQuitting...")
            break
        elif key == ord('r'):
            print("\nResetting tracking...")
            tracker.reset()
            frame_count = 0
        
        frame_count += 1
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print()
    print(f"Demo finished. Processed {frame_count} frames.")


if __name__ == "__main__":
    main()
