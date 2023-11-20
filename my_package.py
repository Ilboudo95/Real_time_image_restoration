
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import tensorflow_hub as hub
from ISR.models import RDN
from PIL import Image
from fastai.vision.all import *
from fastbook import *

import matplotlib.pyplot as plt
import numpy as np
import cv2
from moviepy.editor import VideoFileClip

def resize_image(image, target_dim):
    shape = tf.cast(tf.shape(image)[1:-1], tf.float32)
    short_dim = min(shape)
    scale = target_dim / short_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    image = tf.image.resize(image, new_shape)
    image = tf.image.resize(image, [target_dim, target_dim])
    return image

def get_model(model_url: str, input_resolution: tuple) -> tf.keras.Model:
    inputs = tf.keras.Input((*input_resolution, 3))
    hub_module = hub.KerasLayer(model_url)
    outputs = hub_module(inputs)
    return tf.keras.Model(inputs, outputs)
  
def prediction_operation(image, model):
    image = PILImage.create(image)
    pred, _ , proba_list = model.predict(image)
    prob_list = proba_list.tolist() 
    lab  = ["Deblurring", "Dehazing_Indoor", "Dehazing_Outdoor", "Denoising", "Deraining", "Enhancement", "Super_Resolution"]
    prob_l = {lab[posi]:proba for posi, proba in enumerate(prob_list) if proba>=0.40}
    if len(prob_l.keys()):
        return prob_l
    else:
        return "Clean", None  
    
def resize_imageCV(image, target_dim):
    height, width = image.shape[:2]
    # print(height, width)
    aspect_ratio = width / height
    
    if height < width:
        new_width = target_dim[1]
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = target_dim[0]
        new_width = int(new_height * aspect_ratio)
    image = cv2.resize(image, (new_width, new_height))
    
    return image

def process_imageCV(frame,target_dim=256):
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.astype(np.float32) / 255.0
    frame = resize_imageCV(frame, (target_dim, target_dim))
    return frame

def resize_frame(frame, nouvelle_taille=(256, 256)):
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    frame_resized = cv2.resize(frame, nouvelle_taille, interpolation=cv2.INTER_AREA)
    frame_resized = np.expand_dims(frame_resized, axis=0)
    return frame_resized

def inferCV(frame, model: tf.keras.Model, input_resolution=(256, 256)):
    frame = resize_frame(frame, nouvelle_taille=(256, 256))

    preds = model.predict(frame)
    if isinstance(preds, list):
        preds = preds[-1]
        if isinstance(preds, list):
            preds = preds[-1]

    preds = np.array(preds[0], np.float32)
    final_pred_image = np.array((np.clip(preds, 0.0, 1.0)).astype(np.float32))
    
    return final_pred_image
def prediction_operationCV_im(image, model):
    # image = PILImage.create(image)
    pred, _ , proba_list = model.predict(image)
    prob_list = proba_list.tolist() 
    lab  = ["Deblurring", "Dehazing_Indoor", "Dehazing_Outdoor", "Denoising", "Deraining", "Enhancement", "Super_Resolution"]
    prob_l = {lab[posi]:proba for posi, proba in enumerate(prob_list) if proba>=0.85}
    if len(prob_l.keys()):
        # print("prob_l  ",prob_l)
        return prob_l
    else:
        return "Clean",None
    
def prediction_operationCV(image, model):
    # image = PILImage.create(image)
    pred, _ , proba_list = model.predict(image)
    prob_list = proba_list.tolist() 
    lab  = ["Deblurring", "Dehazing_Indoor", "Dehazing_Outdoor", "Denoising", "Deraining", "Enhancement", "Super_Resolution"]
    prob_l = {lab[posi]:proba for posi, proba in enumerate(prob_list) if proba>=0.85}
    if len(prob_l.keys()):
        # print("prob_l  ",prob_l)
        return prob_l
    else:
        return "Clean",None
    
def conditional_inferenceCV_im(imag,classifier,input_resolution=(256,256)):
    mission = prediction_operationCV_im(imag, classifier)
    image = process_imageCV(imag, input_resolution[0])
    # print("okkk ",image.shape)
    if isinstance(mission, dict):
            tasks = list(mission.keys())
            if len(mission.keys())==1:
                task = tasks[0]
                # print("in")
                print(task)
                if task=="Denoising":  
                    return inferCV(image, Denoising, input_resolution),task
                elif task=="Dehazing_Indoor":      
                    return inferCV(image, Dehazing_Indoor, input_resolution),task
                elif task=="Dehazing_Outdoor":
                    return inferCV(image, Dehazing_Outdoor, input_resolution),task
                elif task=="Deblurring":
                    return inferCV(image, Deblurring, input_resolution),task
                elif task=="Deraining":
                    return inferCV(image, Deraining, input_resolution),task
                elif task=="Enhancement":
                    return inferCV(image, Enhancement, input_resolution),task
                elif task=="Super_Resolution":
                    # print("image unique")
                    print("unique")
                    sr_img = Super_Resolution.predict(imag)
                    return sr_img,task
                else: # clean
                    # print("Clean")
                    return image,"clean"

            else:
                print(tasks)
                poids= list(mission.values())
                predo = []
                for task, poid in zip(tasks,poids):
                    # print(task, poid)
                    # poid = 1
                
                    if task=="Denoising":  
                        predo.append(inferCV(image, Denoising, input_resolution)*poid)
                    elif task=="Dehazing_Indoor":      
                        predo.append(inferCV(image, Dehazing_Indoor, input_resolution)*poid)
                    elif task=="Dehazing_Outdoor":
                        predo.append(inferCV(image, Dehazing_Outdoor, input_resolution)*poid)
                    elif task=="Deblurring":
                        predo.append(inferCV(image, Deblurring, input_resolution)*poid)
                    elif task=="Deraining":
                        predo.append(inferCV(image, Deraining, input_resolution)*poid)
                    elif task=="Enhancement":
                        predo.append(inferCV(image, Enhancement, input_resolution)*poid)
                    elif task=="Super_Resolution":
                        sr_img = Super_Resolution.predict(imag)                   
                        predo.append(sr_img*poid)
        

                somme = predo[0]*0
                for img in predo:
                    somme += img
                    
                mean_img = somme/sum(poids)
                img_array = np.array(mean_img)
                # print("General")
                return img_array,tasks
    else: # clean
                print("Clean")          
                return image,"clean"

def conditional_inferenceCV(imag,classifier,input_resolution=(256,256)):
    mission = prediction_operationCV(imag, classifier)
    image = process_imageCV(imag, input_resolution[0])
    # print("okkk ",image.shape)
    if isinstance(mission, dict):
            tasks = list(mission.keys())
            if len(mission.keys())==1:
                task = tasks[0]
                # print("in")
                print(task)
                if task=="Denoising":  
                    return inferCV(image, Denoising, input_resolution),task
                elif task=="Dehazing_Indoor":      
                    return inferCV(image, Dehazing_Indoor, input_resolution),task
                elif task=="Dehazing_Outdoor":
                    return inferCV(image, Dehazing_Outdoor, input_resolution),task
                elif task=="Deblurring":
                    return inferCV(image, Deblurring, input_resolution),task
                elif task=="Deraining":
                    return inferCV(image, Deraining, input_resolution),task
                elif task=="Enhancement":
                    return inferCV(image, Enhancement, input_resolution),task
                elif task=="Super_Resolutiond":
                    # print("image unique")
                    print("unique")
                    sr_img = Super_Resolution.predict(imag)
                    return sr_img,task
                else: # clean
                    # print("Clean")
                    return image,"clean"

            else:
                poids= list(mission.values())
                predo = []
                for task, poid in zip(tasks,poids):
                    # print(task, poid)
                    # poid = 1
                
                    if task=="Denoising":  
                        predo.append(inferCV(image, Denoising, input_resolution)*poid)
                    elif task=="Dehazing_Indoor":      
                        predo.append(inferCV(image, Dehazing_Indoor, input_resolution)*poid)
                    elif task=="Dehazing_Outdoor":
                        predo.append(inferCV(image, Dehazing_Outdoor, input_resolution)*poid)
                    elif task=="Deblurring":
                        predo.append(inferCV(image, Deblurring, input_resolution)*poid)
                    elif task=="Deraining":
                        predo.append(inferCV(image, Deraining, input_resolution)*poid)
                    elif task=="Enhancement":
                        predo.append(inferCV(image, Enhancement, input_resolution)*poid)
                    elif task=="Super_Resolution":
                        sr_img = Super_Resolution.predict(imag)                   
                        predo.append(sr_img*poid)
        

                somme = predo[0]*0
                for img in predo:
                    somme += img
                    
                mean_img = somme/sum(poids)
                img_array = np.array(mean_img)
                # print("General")
                return img_array,tasks
    else: # clean
                print("Clean")          
                return image,"clean"

def display_side_by_side_image(image_path, process_function, classifier):
    
    # cap = cv2.VideoCapture(video_path)
    frame = cv2.imread(image_path)
    modified_frame = process_function(frame, classifier)

    # Check the heights of both frames
    frame_height = frame.shape[0]
    modified_frame_height = modified_frame[0].shape[0]
        
    # If the heights are different, resize the modified frame to match the original frame's height
    if frame_height != modified_frame_height:
        aspect_ratio = modified_frame_height / frame_height
        modified_width = int(frame.shape[1] * aspect_ratio)
        modified_frame_resized = cv2.resize(modified_frame[0], (modified_width, frame_height))
    else:
        modified_frame_resized = modified_frame[0]
        
        
    # Resize the original frame to match the size of modified_frame_resized
    frame_resized = cv2.resize(frame, (modified_frame_resized.shape[1], modified_frame_resized.shape[0]))
    if modified_frame[1]!="Super_Resolution":
            
        modified_frame_resized = (modified_frame_resized * 255).astype(np.uint8)

    combined_frame = np.hstack((frame_resized,modified_frame_resized))
    return combined_frame,modified_frame[1] 
# ----------------------------------------------------------------

def display_side_by_side(video_path, process_function, classifier, output_path):
    print("\n[Restoration Running] \n" )

    video_path = ""
    video_path = "temp.avi"
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)
    with tf.device('/cpu:0'):
        out = cv2.VideoWriter(output_path,  
                                    cv2.VideoWriter_fourcc(*'mp4v'), 
                                    10, size) 

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            modified_frame = list(process_function(frame, classifier))
            modified_frame[0] = cv2.resize(modified_frame[0], size)
            if modified_frame[1] != "Super_Resolution":
                modified_frame[0] = (modified_frame[0] * 255).astype(np.uint8)

            combined_frame = np.hstack((frame, modified_frame[0]))
            combined_frame = cv2.resize(combined_frame, size)
            with tf.device('/cpu:0'):
                out.write(combined_frame)

            cv2.imshow('video', combined_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            # break
        else:
            break

    cap.release()
    out.release()

    video_clip = VideoFileClip(output_path)
    video_clip.write_videofile(output_path, codec="libx264")

    print(" \n [INFO]: End of video restoration\n")
    cv2.destroyAllWindows()

def gen_frames(video_path, classifier):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)

    while True:
   
        success, frame = cap.read()  
        if success:
            modified_frame = list(conditional_inferenceCV(frame, classifier))
            modified_frame[0] = cv2.resize(modified_frame[0], size)
            if modified_frame[1] != "Super_Resolution":
                modified_frame[0] = (modified_frame[0] * 255).astype(np.uint8)

            bar_color = (0, 0, 255)  # Red bar, you can change the color as needed
            bar_thickness = 5
            frame_with_bar = np.zeros((frame_height, bar_thickness, 3), dtype=np.uint8)
            frame_with_bar[:] = bar_color

            combined_frame = np.hstack((frame, frame_with_bar, modified_frame[0]))
            combined_frame = cv2.resize(combined_frame, size)
            
          
            _, buffer = cv2.imencode('.jpg', combined_frame)
            combined_frame = buffer.tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + combined_frame + b'\r\n')

def gen_frames2(video_path, classifier):
    try:
      camera_ip= int(video_path)  
    except: 
        camera_ip =  video_path
    cap = cv2.VideoCapture(camera_ip)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)

    while True:
   
        success, frame = cap.read()  
        if success:
            modified_frame = list(conditional_inferenceCV(frame, classifier))
            modified_frame[0] = cv2.resize(modified_frame[0], size)
            if modified_frame[1] != "Super_Resolution":
                modified_frame[0] = (modified_frame[0] * 255).astype(np.uint8)

            bar_color = (0, 0, 255)  # Red bar, you can change the color as needed
            bar_thickness = 5
            frame_with_bar = np.zeros((frame_height, bar_thickness, 3), dtype=np.uint8)
            frame_with_bar[:] = bar_color

            combined_frame = np.hstack((frame, frame_with_bar, modified_frame[0]))
            combined_frame = cv2.resize(combined_frame, size)
            
          
            _, buffer = cv2.imencode('.jpg', combined_frame)
            combined_frame = buffer.tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + combined_frame + b'\r\n')

ckpts_path = "modele/weights/"

input_resolution = (256, 256)
print("\n[RUNNING] Restoration models loading ...\n")
Denoising = get_model(model_url= ckpts_path+"Denoising", input_resolution=input_resolution)
Dehazing_Indoor = get_model(model_url= ckpts_path+"Dehazing_Indoor", input_resolution=input_resolution)
Dehazing_Outdoor = get_model(model_url= ckpts_path+"Dehazing_Outdoor", input_resolution=input_resolution)
Deblurring = get_model(model_url= ckpts_path+"Deblurring", input_resolution=input_resolution)
Deraining = get_model(model_url= ckpts_path+"Deraining", input_resolution=input_resolution)
Enhancement = get_model(model_url= ckpts_path+"Enhancement", input_resolution=input_resolution)
Super_Resolution = RDN(weights='noise-cancel')

print("[INFO] : End of loading\n")
