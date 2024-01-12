import torch
import cv2
import numpy as np
from imutils.video import VideoStream
from midas.model_loader import default_models, load_model
import time

# Depth prediction function, outputs a (relative) depth map
@torch.no_grad()
def process(device, model, image, input_size, target_size):
    sample = torch.from_numpy(image).to(device).unsqueeze(0)
    prediction = model.forward(sample)
    prediction = (
        torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=target_size[::-1],
            mode="bicubic",
            align_corners=False,
        )
        .squeeze()
        .cpu()
        .numpy()
    )

    return prediction

# Function that shows the webcam input feed and relative depth map side-by-side
def create_side_by_side(image, depth):
    depth_min = depth.min()
    depth_max = depth.max()
    normalized_depth = 255 * (depth - depth_min) / (depth_max - depth_min)
    normalized_depth *= 3
    right_side = np.repeat(np.expand_dims(normalized_depth, 2), 3, axis=2) / 3
    right_side = cv2.applyColorMap(np.uint8(right_side), cv2.COLORMAP_INFERNO)
    return np.concatenate((image, right_side),1)

# Main code
def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use GPU with NVIDIA CUDA cores if available
    optimize=False
    height=None
    square=False
    model_type= "midas_v21_small_256" # MiDaS model used
    model_path = default_models[model_type]
    model, transform, net_w, net_h = load_model(device,model_path, model_type, optimize, height, square)

    # Open the laptop webcam feed to use as input device
    video = VideoStream(0).start()
    while True:
        start_time = time.time() # Grab time for FPS calculation
        frame=video.read() # Grab a frame from the webcam feed
        if frame is not None:
            original_image_rgb = np.flip(frame,2)
            image = transform({"image": original_image_rgb/255})["image"]
            prediction = process(device, model, image,(net_w, net_h), original_image_rgb.shape[1::-1]) # Calculate Depth Map based on input frame
            original_image_bgr = np.flip(original_image_rgb,2) # Flip image so it isn't mirrored
            content = create_side_by_side(original_image_bgr,prediction) # Put input image and depth map side-by-side

            # Calculate the FPS
            end_time = time.time()
            elapsed_time = end_time - start_time
            fps = 1 / elapsed_time

            # Draw FPS on the frame
            cv2.putText(content, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('MiDaS Depth Estimation', content/255)
            if cv2.waitKey(1) == 27: #Escape key
                break
    cv2.destroyAllWindows()
    video.stop()

if __name__ == "__main__":
    run()