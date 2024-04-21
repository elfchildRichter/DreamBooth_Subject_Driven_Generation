import tensorflow as tf 
import tensorflow.experimental.numpy as tnp
from constants import *

class DreamBoothTrainer(tf.keras.Model):
    
    def __init__(
        self, 
        diffusion_model, 
        vae, 
        noise_scheduler, 
        use_mixed_precision=False, 
        train_text_encoder=None, 
        prior_loss_weight=1.0, 
        max_grad_norm=1.0, 
        max_length=MAX_PROMPT_LENGTH, **kwargs
        ):
        super().__init__(**kwargs)
        self.diffusion_model = diffusion_model
        self.diffusion_model.trainable = True
        self.vae = vae
        self.vae.trainable = False
        self.noise_scheduler = noise_scheduler
        self.use_mixed_precision = use_mixed_precision
        self.max_length = max_length
        self.train_text_encoder = train_text_encoder
        if train_text_encoder:
            self.text_encoder = train_text_encoder
            self.text_encoder.trainable = True
            self.pos_ids = tf.convert_to_tensor([list(range(self.max_length))], dtype=tf.int32)

        self.prior_loss_weight = prior_loss_weight
        self.max_grad_norm = max_grad_norm

    def train_step(self, inputs):
        instance_batch = inputs[0]
        class_batch = inputs[1]

        instance_images = instance_batch["instance_images"]
        instance_texts = instance_batch["instance_texts"]
        class_images = class_batch["class_images"]
        class_texts = class_batch["class_texts"]

        images = tf.concat([instance_images, class_images], axis=0)
        texts = tf.concat([instance_texts, class_texts], axis=0)
        batch_size = tf.shape(images)[0]

        with tf.GradientTape() as tape:
            
            if self.train_text_encoder:
                texts = self.text_encoder([texts, self.pos_ids], training=True)
                
            latents = self.sample_from_encoder_outputs(self.vae(images, training=False))
            latents = latents * 0.18215
            noise = tf.random.normal(tf.shape(latents))
            timesteps = tnp.random.randint(0, self.noise_scheduler.train_timesteps, (batch_size,))
            noisy_latents = self.noise_scheduler.add_noise(tf.cast(latents, noise.dtype), noise, timesteps)
            target = noise
            timestep_embedding = tf.map_fn(lambda t: self.get_timestep_embedding(t), timesteps, fn_output_signature=tf.float32)
            
            model_pred = self.diffusion_model([noisy_latents, timestep_embedding, texts], training=True)
            loss = self.compute_loss(target, model_pred)

            if self.use_mixed_precision:
                loss = self.optimizer.get_scaled_loss(loss)

        trainable_vars = self.diffusion_model.trainable_variables
        if self.train_text_encoder:
            trainable_vars += self.text_encoder.trainable_variables

        gradients = tape.gradient(loss, trainable_vars)
        if self.use_mixed_precision:
            gradients = self.optimizer.get_unscaled_gradients(gradients)
        gradients = [tf.clip_by_norm(g, self.max_grad_norm) for g in gradients]

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return {m.name: m.result() for m in self.metrics}

    def get_timestep_embedding(self, timestep, dim=320, max_period=10000):
        half = dim // 2
        log_max_period = tf.math.log(tf.cast(max_period, tf.float32))
        freqs = tf.math.exp(-log_max_period * tf.range(0, half, dtype=tf.float32) / half)
        args = tf.convert_to_tensor([timestep], dtype=tf.float32) * freqs
        return tf.concat([tf.math.cos(args), tf.math.sin(args)], 0)

    def sample_from_encoder_outputs(self, outputs):
        mean, logvar = tf.split(outputs, 2, axis=-1)
        logvar = tf.clip_by_value(logvar, -30.0, 20.0)
        std = tf.exp(0.5 * logvar)
        sample = tf.random.normal(tf.shape(mean), dtype=mean.dtype)
        return mean + std * sample

    def compute_loss(self, target, model_pred):
        model_pred, model_pred_prior = tf.split(model_pred, num_or_size_splits=2, axis=0)
        target, target_prior = tf.split(target, num_or_size_splits=2, axis=0)
        loss = self.compiled_loss(target, model_pred) 
        prior_loss = self.compiled_loss(target_prior, model_pred_prior) 
        loss = loss + self.prior_loss_weight * prior_loss
        return loss

    # def save_weights(self, filepath, overwrite=True, save_format=None, options=None):
    #     self.diffusion_model.save_weights(
    #         filepath=filepath,
    #         overwrite=overwrite,
    #         save_format=save_format,
    #         options=options,
    #     )
        
    def save_weights(self, ckpt_path_prefix, overwrite=True, save_format=None, options=None):
       
        diffusion_model_path = ckpt_path_prefix + "_d.h5"
        self.diffusion_model.save_weights(
            filepath=diffusion_model_path,
            overwrite=overwrite,
            save_format=save_format,
            options=options,
        )
        self.diffusion_model_path = diffusion_model_path
        
        if self.train_text_encoder:
            text_encoder_model_path = ckpt_path_prefix + "_t.h5"
            self.text_encoder.save_weights(
                filepath=text_encoder_model_path,
                overwrite=overwrite,
                save_format=save_format,
                options=options,
            )
            self.text_encoder_model_path = text_encoder_model_path

    def load_weights(self, filepath, by_name=False, skip_mismatch=False, options=None):
        self.diffusion_model.load_weights(
            filepath=filepath,
            by_name=by_name,
            skip_mismatch=skip_mismatch,
            options=options,
        )
        
        
        
from tensorflow.keras.callbacks import Callback

class CustomSaveCallback(Callback):
    def __init__(self, save_path_prefix, save_best_only=False, monitor="loss", mode="min"):
        super().__init__()
        self.save_path_prefix = save_path_prefix
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode
        self.best = float('inf') if mode == "min" else -float('inf')

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if not self.save_best_only or (self.mode == "min" and current < self.best) or (self.mode == "max" and current > self.best):
            self.model.save_weights(self.save_path_prefix, overwrite=True)
            if self.save_best_only:
                self.best = current