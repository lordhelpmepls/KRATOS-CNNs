#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 14:09:20 2025

@author: marlinmahmud
"""

# TESTED - OK

import cv2
import os

def create_video(image_folder, output_video, frame_rate=30, prefix=""):
    images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png") and img.startswith(prefix)])

    if not images:
        print(f"No images found for prefix {prefix}")
        return

    first_image = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, _ = first_image.shape

    fourcc = cv2.VideoWriter_fourcc(*'H264')
    video_writer = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))

    for img in images:
        img_path = os.path.join(image_folder, img)
        frame = cv2.imread(img_path)
        video_writer.write(frame)  

    video_writer.release()
    print(f"Video {output_video} created successfully!")


density_maps_folder = '/Users/marlinmahmud/Downloads/density_maps_K3'

edge_on_video = 'edge_on_video_K15.mp4'
face_on_video = 'face_on_video_K15.mp4'

create_video(density_maps_folder, face_on_video, prefix="face_on_LMC")
create_video(density_maps_folder, edge_on_video, prefix="edge_on_LMC")