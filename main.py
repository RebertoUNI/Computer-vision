def generate_keypoints(image_shape, dense, step_size=15, scales=[10, 20]):

    if dense: 
        keypoints = []
        rows, cols = image_shape
        # Griglia spaziale
        for y in range(0, rows, step_size):
            for x in range(0, cols, step_size):
                # Iterazione sullo scale-space
                for size in scales:
                    kp = cv2.KeyPoint(float(x), float(y), float(size))
                    keypoints.append(kp)
        return keypoints
    else:
        keypoints, descriptors = sift.detectAndCompute(image, None)