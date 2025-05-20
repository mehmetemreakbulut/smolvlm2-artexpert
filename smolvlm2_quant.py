import torch
from transformers import AutoProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
import quanto

from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

model_path = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
processor = AutoProcessor.from_pretrained(model_path)
model_hf = AutoModelForImageTextToText.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    _attn_implementation="flash_attention_2"
).to("cuda")



# Example image
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")


print("Original Hugging Face model loaded.")
# print(model_hf) # To see the structure

# Apply quantization to linear layers in the vision model and text model parts
# For BLIP, this means targeting self-attention and MLP layers in both vision_model and text_model

def apply_linear_quantization(module, weights_dtype=quanto.qfloat8, activations_dtype=quanto.qfloat8):
    for name, child in module.named_children():
        if isinstance(child, torch.nn.Linear):
            # Replace nn.Linear with a QuantizedLinear with specified weight dtype
            # For static quantization, you'd calibrate with data after this
            # For simplicity, we might use dynamic quantization or just set weights
            quant_linear = quanto.QLinear.from_module(
                child, weights=weights_dtype, activations=activations_dtype # No activation quantization for simplicity
            )
            setattr(module, name, quant_linear)
            # print(f"Replaced {name} with QuantizedLinear")
        else:
            apply_linear_quantization(child, weights_dtype)

print("\nStarting quantization with quanto...")

from copy import deepcopy
full_model = deepcopy(model_hf)

# Quantize vision model parts
'''
if hasattr(model_hf, 'vision_model'):
    apply_linear_quantization(model_hf.vision_model)
    print("Applied quantization to vision_model.")

# Quantize text model parts (text encoder/decoder)
if hasattr(model_hf, 'text_decoder'): # BLIP uses a decoder
    apply_linear_quantization(model_hf.text_decoder)
    print("Applied quantization to text_decoder.")
elif hasattr(model_hf, 'text_model'):
    apply_linear_quantization(model_hf.text_model)
    print("Applied quantization to text_model.")
'''


quanto.quantize(model_hf, weights=quanto.qint8, activations=quanto.qint2)


with quanto.Calibration(momentum=0.9):
    samples = processor_hf(images=image, return_tensors="pt")
    model_hf.generate(**samples)


# After defining the quantization strategy (e.g., replacing modules),
# for static quantization, you would calibrate the model:
# quanto.calibrate(model_hf, calibration_data_generator_function)
# For this example, we'll assume dynamic quantization or that calibration
# would be done if activations were also quantized.
# Freezing is important to make the model use quantized ops.
quanto.freeze(model_hf)
print("Quanto model frozen.")



#also quantize activations


# Check the dtype of a quantized layer's weight
# The path to the layer will depend on the model architecture.
# For BLIP, let's try to find a q_proj or similar.
# Example: model_hf.vision_model.encoder.layers[0].self_attn.q_proj.weight
# This path is from your original example and might not exist in BLIP as is.
# Let's find a linear layer in BLIP to check:


# To verify, you can try running inference with the quantized PyTorch model
# (ensure inputs are on the correct device if you moved the model to GPU)

inputs_hf = processor_hf(images=image, return_tensors="pt")
import time
import psutil
import os


import matplotlib.pyplot as plt
#sample an activation tensor from the quantized model, putting a hook to model_hf.vision_model.encoder.layers[0].self_attn
def sample_activation(module, input, output):
    #plot the output tensor
    tensor = output.detach().cpu().numpy()
    #plot the tensor as a histogram
    tensor_flat = tensor.flatten()
    #plot the histogram of the tensor
    plt.hist(tensor_flat)
    print(tensor_flat)
    plt.show()

model_hf.vision_model.encoder.layers[0].mlp.register_forward_hook(sample_activation)



start_time = time.time()
outputs_hf_quantized = model_hf.generate(**inputs_hf)
caption_quantized = processor_hf.decode(outputs_hf_quantized[0], skip_special_tokens=True)
print(f"Caption from Quanto-quantized PyTorch model: {caption_quantized}")
end_time = time.time()
#calculate the memory usage of the quantized  model and the original model
import psutil
import os
print(f"Memory usage of quantized model: {psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024:.2f} MB")
print(f"Time taken: {end_time - start_time} seconds")

# not quantized model
start_time = time.time()
outputs_hf = full_model.generate(**inputs_hf)
caption = processor_hf.decode(outputs_hf[0], skip_special_tokens=True)
print(f"Caption from original PyTorch model: {caption}")
end_time = time.time()
#calculate the memory usage of the original model   
print(f"Memory usage of original model: {psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024:.2f} MB")
print(f"Time taken: {end_time - start_time} seconds")





