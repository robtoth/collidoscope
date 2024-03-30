from diffusers import StableDiffusionPipeline
import torch

model_id = "CompVis/stable-diffusion-v1-4"

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True)
pipe = pipe.to(device)

def generate_image(prompt):
    image = pipe(prompt, guidance_scale=7.5).images[0]
    return image

prompt = "A colorful kaleidoscope pattern."
image = generate_image(prompt)
image.save("kaleidoscope.png")