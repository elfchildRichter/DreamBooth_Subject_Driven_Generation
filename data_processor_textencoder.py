import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers.experimental.preprocessing import CenterCrop, Rescaling, RandomFlip
from keras_cv.models.stable_diffusion.clip_tokenizer import SimpleTokenizer
from keras_cv.models.stable_diffusion.text_encoder import TextEncoder
import itertools
from typing import Callable, Dict, List, Tuple
from constants import *

POS_IDS = tf.convert_to_tensor([list(range(MAX_PROMPT_LENGTH))], dtype=tf.int32)
AUTO = tf.data.AUTOTUNE


class DataProcessor:
    def __init__(
        self,
        instance_image_paths: List[str],
        class_image_paths: List[str],
        unique_id: str,
        class_category: str,
        train_text_encoder: bool,
        img_height: int = RESOLUTION,
        img_width: int = RESOLUTION,
        batch_size: int = BATCH_SIZE,
    ):
        self.instance_image_paths = instance_image_paths
        self.class_image_paths = class_image_paths
        self.unique_id = unique_id
        self.class_category = class_category
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.tokenizer = SimpleTokenizer()
        self.train_text_encoder = train_text_encoder
        
        if not self.train_text_encoder:
            self.text_encoder = TextEncoder(MAX_PROMPT_LENGTH)
            
        self.augmenter = Sequential([
            CenterCrop(height=self.img_height, width=self.img_width),
            RandomFlip("horizontal_and_vertical"),
            Rescaling(scale=1.0 / 127.5, offset=-1)
        ])

    def _get_captions(self, num_instance_images: int, num_class_images: int) -> Tuple[List, List]:
        instance_caption = f"a photo of {self.unique_id} {self.class_category}"
        instance_captions = [instance_caption] * num_instance_images
        class_caption = f"a photo of {self.class_category}"
        class_captions = [class_caption] * num_class_images
        return instance_captions, class_captions

    def _tokenize_text(self, caption: str) -> np.ndarray:
        tokens = self.tokenizer.encode(caption)
        tokens = tokens + [PADDING_TOKEN] * (MAX_PROMPT_LENGTH - len(tokens))
        return np.array(tokens)

    def _tokenize_captions(self, instance_captions: List[str], class_captions: List[str]) -> np.ndarray:
        tokenized_texts = np.empty((len(instance_captions) + len(class_captions), MAX_PROMPT_LENGTH))
        for i, caption in enumerate(itertools.chain(instance_captions, class_captions)):
            tokenized_texts[i] = self._tokenize_text(caption)
            
        return tokenized_texts

    def _embed_captions(self, tokenized_texts: np.ndarray) -> np.ndarray:
        gpus = tf.config.list_logical_devices("GPU")
        with tf.device(gpus[0].name):
            embedded_text = self.text_encoder([tf.convert_to_tensor(tokenized_texts), POS_IDS], training=False).numpy()

        del self.text_encoder 
        return embedded_text

    def _collate_instance_image_paths(self, instance_image_paths: List[str], class_image_paths: List[str]) -> List:
        new_instance_image_paths = []
        for index in range(len(class_image_paths)):
            instance_image = instance_image_paths[index % len(instance_image_paths)]
            new_instance_image_paths.append(instance_image)
            
        return new_instance_image_paths

    def _process_image(self, image_path: tf.Tensor, text: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        image = tf.io.read_file(image_path)
        image = tf.io.decode_png(image, 3)
        image = tf.image.resize(image, (self.img_height, self.img_width))
        return image, text

    def _apply_augmentation(self, image_batch: tf.Tensor, text_batch: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        return self.augmenter(image_batch), text_batch

    def _prepare_dict(self, instance_only=True) -> Callable:
        def fn(image_batch, texts) -> Dict[str, tf.Tensor]:
            if instance_only:
                batch_dict = {"instance_images": image_batch, "instance_texts": texts}
                return batch_dict
            else:
                batch_dict = {"class_images": image_batch, "class_texts": texts}
                return batch_dict
        return fn

    def _assemble_dataset(self, image_paths: List[str], texts: np.ndarray, instance_only=True) -> tf.data.Dataset:
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, texts))
        dataset = dataset.map(self._process_image, num_parallel_calls=AUTO)
        dataset = dataset.shuffle(self.batch_size * 10, reshuffle_each_iteration=True)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.map(self._apply_augmentation, num_parallel_calls=AUTO)
        prepare_dict_fn = self._prepare_dict(instance_only=instance_only)
        dataset = dataset.map(prepare_dict_fn, num_parallel_calls=AUTO)
        return dataset

    def prepare_datasets(self) -> tf.data.Dataset:
        instance_captions, class_captions = self._get_captions(len(self.instance_image_paths), len(self.class_image_paths))
        text_batch = self._tokenize_captions(instance_captions, class_captions)
        if not self.train_text_encoder:
            print("Embedding captions via TextEncoder...")
            text_batch = self._embed_captions(text_batch)

        print("Assembling instance and class datasets...")
        instance_dataset = self._assemble_dataset(self.instance_image_paths, text_batch[: len(self.instance_image_paths)])
        class_dataset = self._assemble_dataset(self.class_image_paths, text_batch[len(self.instance_image_paths) :], 
                                               instance_only=False)

        train_dataset = tf.data.Dataset.zip((instance_dataset, class_dataset))
        return train_dataset.prefetch(AUTO)