
# This file was generated by the Tkinter Designer by Parth Jadhav
# https://github.com/ParthJadhav/Tkinter-Designer


from pathlib import Path

# from tkinter import *
# Explicit imports to satisfy Flake8
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage,messagebox
from Faces import start

from train_data import train

import math
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder

import cv2
import numpy as np
from itertools import chain


OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path("./assets")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")
unknown_dir = os.path.join(BASE_DIR, "unknown")


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)


window = Tk()

window.geometry("800x500")
window.configure(bg = "#B0C1FA")

def onTrain():
    messagebox.showinfo("START", "Training KNN classifier...")
    train(image_dir, os.path.join(BASE_DIR, "trained_knn_model.clf"), n_neighbors=2)
    messagebox.showinfo("SUCCESS", "The model has been successfully trained!")

canvas = Canvas(
    window,
    bg = "#B0C1FA",
    height = 500,
    width = 800,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)
canvas.create_rectangle(
    0.0,
    0.0,
    800.0,
    500.0,
    fill="#B0C1FA",
    outline="")

canvas.create_rectangle(
    0.0,
    452.0,
    800.0,
    500.0,
    fill="#6A7DF5",
    outline="")

canvas.create_text(
    266.0,
    81.0,
    anchor="nw",
    text="AI Face Recognition",
    fill="#000000",
    font=("Prompt Regular", 30 * -1)
)

button_image_1 = PhotoImage(
    file=relative_to_assets("button_1.png"))
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: onTrain(),
    relief="flat"
)
button_1.place(
    x=97.0,
    y=206.0,
    width=237.0,
    height=102.0
)

button_image_2 = PhotoImage(
    file=relative_to_assets("button_2.png"))

button_2 = Button(
    image=button_image_2,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: start(),
    relief="flat"
)
button_2.place(
    x=459.0,
    y=206.0,
    width=287.0,
    height=102.0
)
window.resizable(False, False)
window.mainloop()
