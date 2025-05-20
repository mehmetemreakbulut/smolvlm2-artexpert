
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

# Load the model
model, processor = load("mlx-community/SmolVLM2-2.2B-Instruct-mlx")
config = load_config("mlx-community/SmolVLM2-2.2B-Instruct-mlx")

# Prepare input
image = ["http://images.cocodataset.org/val2017/000000039769.jpg"]
prompt = "Describe this image."

# Apply chat template
formatted_prompt = apply_chat_template(
    processor, config, prompt, num_images=1
)

# Generate output
#output = generate(model, processor, formatted_prompt, image)
#print(output)

#check the parameter dtypes in the model
import torch

#model.vision_model.encoder.quantize()

print(model.vision_model.encoder.layers[0].self_attn.q_proj.weight.dtype)
