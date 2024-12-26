import os
import numpy as np
from PIL import Image
from tkinter import Tk, filedialog

def generate_face_fast(equirectangular, direction, face_size):
    height, width, _ = equirectangular.shape

    u, v = np.meshgrid(
        np.linspace(-1, 1, face_size, dtype=np.float32),
        np.linspace(-1, 1, face_size, dtype=np.float32)
    )

    if direction == "front":
        dx, dy, dz = u, v, 1
    elif direction == "back":
        dx, dy, dz = -u, v, -1
    elif direction == "left":
        dx, dy, dz = -1, v, -u
    elif direction == "right":
        dx, dy, dz = 1, v, u
    elif direction == "top":
        dx, dy, dz = u, 1, -v
    elif direction == "bottom":
        dx, dy, dz = u, -1, v
    else:
        raise ValueError("Invalid direction")

    length = np.sqrt(dx**2 + dy**2 + dz**2)
    dx /= length
    dy /= length
    dz /= length

    lon = np.arctan2(dx, dz)
    lat = np.arcsin(dy)

    px = ((lon + np.pi) / (2 * np.pi) * width).astype(np.int32)
    py = ((1 - (lat + np.pi / 2) / np.pi) * height).astype(np.int32)

    px = np.clip(px, 0, width - 1)
    py = np.clip(py, 0, height - 1)

    face = equirectangular[py, px]

    return face

def hdr_to_cubemap(input_path, output_path, face_size=1024):
    hdr_image = Image.open(input_path).convert("RGB")
    equirectangular = np.array(hdr_image)

    os.makedirs(output_path, exist_ok=True)

    face_mappings = {
        "SkyboxBk": ("back", None),
        "SkyboxDn": ("bottom", -90),
        "SkyboxFt": ("front", None),
        "SkyboxLf": ("left", None),
        "SkyboxRt": ("right", None),
        "SkyboxUp": ("top", 90)
    }

    base_filename = os.path.splitext(os.path.basename(input_path))[0]

    for face_name, (direction, rotation) in face_mappings.items():
        print(f"Generating {face_name} face...")
        face = generate_face_fast(equirectangular, direction, face_size)
        face_image = Image.fromarray(face).rotate(180, expand=True)
        if face_name in ["SkyboxLf", "SkyboxRt"]:
            face_image = face_image.transpose(Image.FLIP_LEFT_RIGHT)

        if rotation:
            face_image = face_image.rotate(rotation, expand=True)

        output_filename = f"{face_name}_{base_filename}.png"
        face_image.save(os.path.join(output_path, output_filename))
        print(f"Saved {face_name} face to {os.path.join(output_path, output_filename)}")

def process_folder(folder_path, output_folder, face_size=1024):
    for filename in os.listdir(folder_path):
        input_path = os.path.join(folder_path, filename)
        output_path = os.path.join(output_folder, os.path.splitext(filename)[0])
        print(f"Processing: {filename}")
        hdr_to_cubemap(input_path, output_path, face_size)

if __name__ == "__main__":
    root = Tk()
    root.withdraw()

    print("Select input folder containing HDRI files:")
    input_folder = filedialog.askdirectory(title="Select Input Folder")
    if not input_folder:
        print("No input folder selected. Exiting.")
        exit()

    print("Select output folder to save cubemaps:")
    output_folder = filedialog.askdirectory(title="Select Output Folder")
    if not output_folder:
        print("No output folder selected. Exiting.")
        exit()

    process_folder(input_folder, output_folder)
