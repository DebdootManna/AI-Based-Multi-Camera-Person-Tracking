import cv2
import numpy as np
import os

def generate_test_video(output_path, num_frames=100, width=640, height=480, num_objects=3):
    """Generate a test video with moving rectangles."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 10.0, (width, height))
    
    # Initialize random positions and velocities for objects
    objects = []
    for _ in range(num_objects):
        x = np.random.randint(50, width-100)
        y = np.random.randint(50, height-100)
        dx = np.random.uniform(-3, 3)
        dy = np.random.uniform(-3, 3)
        color = (np.random.randint(0, 255), 
                 np.random.randint(0, 255), 
                 np.random.randint(0, 255))
        width_obj = np.random.randint(30, 100)
        height_obj = np.random.randint(50, 150)
        objects.append({
            'x': x, 'y': y, 'dx': dx, 'dy': dy,
            'color': color,
            'width': width_obj,
            'height': height_obj
        })
    
    for _ in range(num_frames):
        frame = np.ones((height, width, 3), dtype=np.uint8) * 240  # Light gray background
        
        # Update and draw each object
        for obj in objects:
            # Update position
            obj['x'] += obj['dx']
            obj['y'] += obj['dy']
            
            # Bounce off walls
            if obj['x'] <= 0 or obj['x'] + obj['width'] >= width:
                obj['dx'] *= -1
                obj['x'] = max(0, min(obj['x'], width - obj['width']))
            if obj['y'] <= 0 or obj['y'] + obj['height'] >= height:
                obj['dy'] *= -1
                obj['y'] = max(0, min(obj['y'], height - obj['height']))
            
            # Draw the object
            cv2.rectangle(frame,
                        (int(obj['x']), int(obj['y'])),
                        (int(obj['x'] + obj['width']), int(obj['y'] + obj['height'])),
                        obj['color'], -1)
        
        out.write(frame)
    
    out.release()
    print(f"Generated test video: {output_path}")

if __name__ == "__main__":
    # Create test_videos directory if it doesn't exist
    os.makedirs("test_videos", exist_ok=True)
    
    # Generate two synchronized test videos with slightly different views
    generate_test_video("test_videos/camera1.mp4", num_objects=3)
    generate_test_video("test_videos/camera2.mp4", num_objects=3)
