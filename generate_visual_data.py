from diffusers import FlaxStableDiffusionXLPipeline
from flax.jax_utils import replicate
import argparse



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="Comp")
    args = parser.parse_args()
    pipe, params = FlaxStableDiffusionXLPipeline.from_pretrained(
        "latent-consistency/lcm-sdxl",
    )
    print(pipe)
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")
    



if __name__ == "__main__":
    main()
