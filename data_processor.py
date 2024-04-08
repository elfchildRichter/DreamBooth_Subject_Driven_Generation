import numpy as np
import tensorflow as tf
import keras_cv
from tensorflow.keras.layers.experimental.preprocessing import CenterCrop, Rescaling
from tensorflow.keras import Sequential
from constants import *

class DataProcessor:
    def __init__(self, instance_image_paths, class_image_paths, unique_id=UNIQUE_ID, class_label=CLASS_LABEL):
        self.instance_image_paths = instance_image_paths
        self.class_image_paths = class_image_paths
        self.unique_id = unique_id
        self.class_label = class_label
        self.tokenizer = keras_cv.models.stable_diffusion.SimpleTokenizer()
        self.max_prompt_length = MAX_PROMPT_LENGTH
        self.padding_token = PADDING_TOKEN
        self.resolution = RESOLUTION
        self.auto = tf.data.AUTOTUNE
        self.embedded_text = self._embed_texts()

    def _extend_paths(self):
        new_instance_image_paths = []
        for index in range(len(self.class_image_paths)):
            instance_image = self.instance_image_paths[index % len(self.instance_image_paths)]
            new_instance_image_paths.append(instance_image)
        return new_instance_image_paths

    def _process_text(self, caption):
        tokens = self.tokenizer.encode(caption)
        tokens = tokens + [self.padding_token] * (self.max_prompt_length - len(tokens))
        return np.array(tokens)

    def _embed_texts(self):
        instance_prompts = [f"a photo of {self.unique_id} {self.class_label}"] * len(self._extend_paths())
        class_prompts = [f"a photo of {self.class_label}"] * len(self.class_image_paths)
        all_prompts = instance_prompts + class_prompts
        tokenized_texts = np.array([self._process_text(caption) for caption in all_prompts])
        
        pos_ids = tf.convert_to_tensor([list(range(self.max_prompt_length))], dtype=tf.int32)
        text_encoder = keras_cv.models.stable_diffusion.TextEncoder(self.max_prompt_length)
        
        with tf.device('CPU:0'):
            embedded_text = text_encoder([tf.convert_to_tensor(tokenized_texts, dtype=tf.int32), pos_ids], training=False).numpy()
        del text_encoder
        return embedded_text

    def _create_augmenter(self):
        augmenter = Sequential([
            CenterCrop(height=self.resolution, width=self.resolution),
            keras_cv.layers.RandomFlip(),
            Rescaling(scale=1.0 / 127.5, offset=-1),
        ])
        return augmenter

    def _process_image(self, image_path, tokenized_text):
        image = tf.io.read_file(image_path)
        image = tf.io.decode_png(image, channels=3)
        image = tf.image.resize(image, (self.resolution, self.resolution))
        return image, tokenized_text

    def _apply_augmentation(self, image_batch, embedded_tokens):
        augmenter = self._create_augmenter()
        return augmenter(image_batch), embedded_tokens

    def prepare_dict(self, instance_only=True):
        def fn(image_batch, embedded_tokens):
            if instance_only:
                return {"instance_images": image_batch, "instance_embedded_texts": embedded_tokens}
            else:
                return {"class_images": image_batch, "class_embedded_texts": embedded_tokens}
        return fn

    def assemble_dataset(self, instance_only=True, batch_size=1):
        if instance_only:
            image_paths = self._extend_paths()
            embedded_texts = self.embedded_text[:len(image_paths)]
        else:
            image_paths = self.class_image_paths
            embedded_texts = self.embedded_text[len(self._extend_paths()):]

        dataset = tf.data.Dataset.from_tensor_slices((image_paths, embedded_texts))
        dataset = dataset.map(self._process_image, num_parallel_calls=self.auto)
        dataset = dataset.batch(batch_size)
        dataset = dataset.map(self._apply_augmentation, num_parallel_calls=self.auto)
        prepare_dict_fn = self.prepare_dict(instance_only=instance_only)
        dataset = dataset.map(prepare_dict_fn, num_parallel_calls=self.auto)
        return dataset
