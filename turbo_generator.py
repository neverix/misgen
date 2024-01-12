# did I have to name it that? yes I did
from diffusers import FlaxEulerDiscreteScheduler, FlaxAutoencoderKL
from diffusers import FlaxStableDiffusionXLPipeline
from flax.training.common_utils import shard
from matplotlib import pyplot as plt
from flax.jax_utils import replicate
from flax import linen as nn
import jax.numpy as jnp
import jax


def create_key(seed=0):
    return jax.random.PRNGKey(seed)

# for binding to the VAE
def decode(self, *args, **kwargs):
    return self.decode(*args, **kwargs)

class TurboGenerator(object):
    def __init__(self,
                 path = "nev/sdxl-turbo-pt",
                 vae_scheduler_path = "stabilityai/stable-diffusion-xl-base-1.0",
                 dtype = jnp.bfloat16):
        if vae_scheduler_path is None:
            vae_scheduler_path = path
        self.pipe, params = FlaxStableDiffusionXLPipeline.from_pretrained(
            path,
            from_pt=True,
            dtype=jnp.float32
        )
        params = jax.tree_util.tree_map(lambda x: x.astype(dtype), params)
        self.p_params = replicate(params)
    
        _, self.scheduler_state = FlaxEulerDiscreteScheduler.from_pretrained(
            vae_scheduler_path,
            subfolder="scheduler"  # mad at them for not calling it scheuler
        )
        _, vae_state = FlaxAutoencoderKL.from_pretrained(
            vae_scheduler_path,
            subfolder="vae"
        )
        self.p_vae_state = replicate(vae_state)

        self.decode = nn.apply(decode, self.pipe.vae)

    def generate(self, prompt, imgs_per_device = 4, height = 512, width = 512):
        prompt = "a cat"
        prompts = [prompt] * jax.device_count() * imgs_per_device
        prompt_ids = self.pipe.prepare_inputs(prompts)
        prompt_ids = shard(prompt_ids)
        prompt_embeds, pooled_embeds = jax.pmap(
            self.pipe.get_embeddings)(prompt_ids, self.p_params)
        add_time_ids = self.pipe._get_add_time_ids(
            (height, width), (0, 0), (height, width), imgs_per_device, dtype=prompt_embeds.dtype
        )
        latents_shape = (
            imgs_per_device,
            self.pipe.unet.config.in_channels,
            height // self.pipe.vae_scale_factor,
            width // self.pipe.vae_scale_factor,
        )
        latents = jax.pmap(lambda x: jax.random.normal(x, shape=latents_shape, dtype=jnp.float32))(rng)

        added_cond_kwargs = {"text_embeds": pooled_embeds, "time_ids": replicate(add_time_ids)}
        noise_pred = jax.pmap(self.pipe.unet.apply)(
            {"params": self.p_params["unet"]},
            jnp.array(latents),
            replicate(jnp.array([self.pipe.scheduler.config.num_train_timesteps - 1] * imgs_per_device, dtype=jnp.int32)),
            prompt_embeds,
            added_cond_kwargs,
        ).sample

        denoised = latents - noise_pred
        decoded = jax.pmap(self.decode)(
            {"params": self.p_vae_state},
            denoised * self.scheduler_state.init_noise_sigma / self.pipe.vae.scaling_factor).sample
        return decoded.transpose(1, 2, 0) / 2 + 0.5


if __name__ == "__main__":
    tg = TurboGenerator()
    result = tg.generate("a cat")
    plt.imshow(result[0, 0].transpose(1, 2, 0) / 2 + 0.5)
    plt.show()
