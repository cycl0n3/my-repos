import cv2

def main():
    # Read the image from file
    image = cv2.imread('./images/img02.jpeg')

    # Check if the image was successfully loaded
    if image is None:
        print("Error: Could not read the image.")
        return
    
    smoothed_image = cv2.medianBlur(image, 5)

    # Display the image in a window
    cv2.imshow('Origial Image', image)
    cv2.imshow('Smoothed Image', smoothed_image)

    while True:
        # Wait for a key press or window close event
        key = cv2.waitKey(1)
        
        # Check if the window is still open
        if cv2.getWindowProperty('Origial Image', cv2.WND_PROP_VISIBLE) < 1 or \
          cv2.getWindowProperty('Smoothed Image', cv2.WND_PROP_VISIBLE) < 1:
            break

        # Exit loop if 'q' key is pressed
        if key == ord('q'):
            break

    # Close all OpenCV windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
