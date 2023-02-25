import cv2 as cv
import mediapipe as mp
import numpy as np

mp_selfie = mp.solutions.selfie_segmentation

cap = cv.VideoCapture(0)

with mp_selfie.SelfieSegmentation(model_selection=0) as model:
    while cap.isOpened():
        ret, frame = cap.read()

        # apply segmentation
        frame.flags.writeable = False
        #frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        res = model.process(frame)
        frame.flags.writeable = True

        cv.imshow("Selfie Segmentation", frame)

        if cv.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv.destroyAllWindows()

#probabilities of every pixel's being ourselves (selfie)
print(res.segmentation_mask)

import matplotlib.pyplot as plt

#make background black
frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
background = np.zeros(frame.shape, dtype=np.uint8)
mask = np.stack((res.segmentation_mask,)*3, axis=-1) > 0.5

segmented_image = np.where(mask, frame, background)

plt.imshow(segmented_image)

#make background blur
segmented_image = np.where(mask, frame, cv.blur(frame, (40,40)))

plt.imshow(segmented_image)

plt.show()

### GRADIO APP
# comment every code line above to make gradio app work
# if you work on notebook it is going to be fine to just run the cell contains below code

import gradio as gr

def segment(image):
    with mp_selfie.SelfieSegmentation(model_selection=0) as model:
        #frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        res = model.process(image)
        mask = np.stack((res.segmentation_mask,)*3, axis=-1) > 0.5
        return np.where(mask, image, cv.blur(image, (40,40)))

webcam = gr.inputs.Image(shape=(640, 480), source="webcam")

webapp = gr.Interface(fn=segment, inputs=webcam, outputs="image")

webapp.launch()