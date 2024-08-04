import cv2
import numpy as np

def main():
    # Read the image from file
    image = cv2.imread('./images/img03.jpg', cv2.IMREAD_GRAYSCALE)

    # Check if the image was successfully loaded
    if image is None:
        print("Error: Could not read the image.")
        return
    
    #kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    kernel = np.array([
      [0.1, 0.2, 0.1],
      [0.1, 0.5, 0.1],
      [0.1, 0.2, 0.1]
    ], dtype=np.float32)

    kernel /= np.sum(kernel)
    
    smoothed_image = cv2.filter2D(image, -1, kernel)

    difference = cv2.absdiff(image, smoothed_image)

    # Display the image in a window
    cv2.imshow('Origial Image', image)
    cv2.imshow('Smoothed Image', smoothed_image)
    cv2.imshow('Difference Image', difference)

    while True:
        # Wait for a key press or window close event
        key = cv2.waitKey(1)
        
        # Check if the window is still open
        if cv2.getWindowProperty('Origial Image', cv2.WND_PROP_VISIBLE) < 1 or \
          cv2.getWindowProperty('Smoothed Image', cv2.WND_PROP_VISIBLE) < 1 or \
          cv2.getWindowProperty('Smoothed Image', cv2.WND_PROP_VISIBLE) < 1:
            break

        # Exit loop if 'q' key is pressed
        if key == ord('q'):
            break

    # Close all OpenCV windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
