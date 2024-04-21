import os
import numpy as np
import math
import keras
from tqdm import tqdm
from imutils import paths
import PIL
import hashlib
import matplotlib.pyplot as plt


def load_images(image_paths):
    return [np.array(keras.utils.load_img(path)) for path in image_paths]

def plot_images(images, title=None):
    rows = math.ceil(len(images) / 6) 
    plt.figure(figsize=(18, rows * 3)) 
    for i in range(len(images)):
        ax = plt.subplot(rows, 6, i + 1) 
        if title is not None:
            plt.title(title)
        plt.imshow(images[i])
        plt.axis("off")
    plt.tight_layout()
    plt.show()

def generate_and_save_images(save_folder, model, prompt, n_imgs_gen, batch_size=3, steps=None, ugs=None):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for _ in tqdm(range(n_imgs_gen)):
        kwargs = {'prompt': prompt, 'batch_size': batch_size}
        if steps is not None:
            kwargs['num_steps'] = steps
        if ugs is not None:
            kwargs['unconditional_guidance_scale'] = ugs
        images = model.text_to_image(**kwargs)
        
        idx = np.random.choice(len(images))
        selected_image = PIL.Image.fromarray(images[idx])
        hash_image = hashlib.sha1(selected_image.tobytes()).hexdigest()
        image_filename = os.path.join(save_folder, f"{hash_image}.jpg")
        selected_image.save(image_filename)
        
def plot_training_results(training_results):
    colors = ['darkred', 'darkorange', 'darkgreen', 'darkblue']
    plt.figure(figsize=(9, 5))
    for (label, data), color in zip(training_results.items(), colors):
        plt.plot(data["epochs"], data["loss"], marker='o', linestyle='-', label=label, color=color)

    plt.title('Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(range(1, 6))
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.tight_layout()