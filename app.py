import streamlit as st
from PIL import Image
import numpy as np
import cv2
import torch
from torcht import Age_model_Resnet
import torch.nn as nn
from torchvision import models




st.divider()
col1, col2 = st.columns(2)

with col1:
    st.subheader("Original")
    picture = st.camera_input("Take a picture")
    

img_tensor = None


#cái này con tham khảo AI
def cropimage(image):
    h, w = image.shape[:2]
    detector = cv2.FaceDetectorYN.create(
        "face_detection_yunet.onnx",
        "",
        (w, h),
        score_threshold=0.9
    )
    _, faces = detector.detect(image)

    if faces is None:
        print("No faces detected")
        return None
    

    x, y, bw, bh = faces[0][:4].astype(int)
    
    cx, cy = x + bw // 2, y + bh // 2
    largest_side = max(bw,bh)
    half = largest_side // 2

    x0, y0 = cx - half, cy - half
    x1, y1 = x0 + 2 * half, y0 + 2 * half 

    if x0 < 0:
        x0 = 0
    if y0 < 0:
        y0 = 0

    
    return image[y0:y1 ,x0:x1]




with col2:
    if picture is not None:
        pil_image = Image.open(picture).convert('RGB')
        np_image = np.array(pil_image)  
        bgr_img = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)     
        cropped = cropimage(bgr_img)   

        if cropped is not None:
            cropped = cv2.resize(cropped, (224,224), interpolation=cv2.INTER_AREA)
            
            st.image(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB), caption="My photo")

            img_tensor = torch.tensor(cropped)
            img_tensor = img_tensor.permute(2,0,1)
        else:
            st.write("No face detected — try again.")




#MODEL
if img_tensor is not None:
    st.write(img_tensor.shape)

resnet_model = models.resnet18(weights=None)
model = Age_model_Resnet(
    pretrained_model=resnet_model
)

device = torch.device("cpu")

state_dict = torch.load(
    "resnet_model.pth",
    map_location=device,
    weights_only=True
)

model.load_state_dict(state_dict)


model.eval()


if img_tensor is not None:
    print()
    test_img = img_tensor.float() / 255.
    test_img = test_img.unsqueeze(0)


    with torch.no_grad():
        y_pred = model(test_img)
        y_pred = y_pred.numpy() * 80

    age = y_pred
    st.write("Predicted age:", round(age))
    