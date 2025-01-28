#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 14:09:15 2025

@author: marlinmahmud
"""

### TESTED - OK


import cv2
import os

def create_video(image_folder, output_video, frame_rate=30, prefix=""):
    images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png") and img.startswith(prefix)])

    if not images:
        print(f"No images found for prefix {prefix}")
        return

    # Read first image
    first_image = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, _ = first_image.shape

    # Create video writer object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  
    video_writer = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))

    for img in images:
        img_path = os.path.join(image_folder, img)
        frame = cv2.imread(img_path)
        video_writer.write(frame)  

    # Release video writer
    video_writer.release()
    print(f"Video {output_video} created successfully!")


density_maps_folder = '/Users/marlinmahmud/Downloads/density_maps_green'

face_on_video = 'face_on_video.mp4'
edge_on_video = 'edge_on_video.mp4'

# Create videos for face-on and edge-on images
create_video(density_maps_folder, face_on_video, prefix="face_on")
create_video(density_maps_folder, edge_on_video, prefix="edge_on")