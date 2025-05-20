from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from PIL import Image
import io
import os
import uuid # For unique temporary filenames
import json # Added for parsing JSON and for SSE
import time # Added for simulating streaming delay (optional)
import logging # Import the logging module
import mlx_vlm

# MLX imports
from mlx_vlm import load, generate, stream_generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# --- Model Loading (do this once globally) ---
print("Loading MLX model and processor...")
# Note: If your model path is different or you want it configurable, adjust here.
MODEL_PATH = "mlx-community/SmolVLM2-2.2B-Instruct-mlx"
try:
    model, processor = load(MODEL_PATH)
    config = load_config(MODEL_PATH)
    print("Model and processor loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    # Depending on the severity, you might want to exit or handle this gracefully
    model, processor, config = None, None, None 
# --- End Model Loading ---

# Define a directory for temporary image uploads
UPLOAD_FOLDER = 'temp_uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

SYSTEM_PROMPT = """You are SmolArtExpert, an insightful and slightly witty art historian.
The user will show you an image of an artwork (painting, sculpture, architecture, etc.).
Your primary goal is to analyze it.

Follow these steps for your analysis:
1.  **Identify the Artwork:**
    *   Describe what you see in general terms.
    *   Classify the artwork (e.g., oil painting, marble sculpture, Gothic cathedral detail).
    *   If you confidently recognize the specific artwork, artist, or architect, state it. If not, do not speculate.
2.  **Artistic Style and Period:**
    *   Identify the dominant artistic style(s) (e.g., Renaissance, Baroque, Impressionism, Modernism, Art Deco, Gothic).
    *   Briefly explain the key characteristics of this style as visible in the artwork.
    *   Mention the historical period/era associated with the style.
3.  **Key Features and Symbolism (if applicable):**
    *   Point out 2-3 prominent artistic elements or techniques used (e.g., use of color, light and shadow, composition, brushstrokes, architectural features like arches or columns).
    *   If there's clear symbolism or common interpretations associated with elements in the artwork or its style, briefly mention them.
4.  **Historical Context Snippet:**
    *   Provide a concise, interesting historical fact or tidbit related to the artwork's style, period, or a typical theme it represents. Keep it engaging!
5.  **Witty Observation (Optional & Brief):**
    *   If appropriate, add a very short, lighthearted (but still respectful) observation or comment. For example, "One can only imagine the patience required for those tiny brushstrokes!" or "This building certainly wasn't designed for a quick weekend project."

**Output Format:**
*   Present your analysis in well-structured bullet points under clear headings (e.g., "Artwork Identification:", "Style & Period:", "Key Features:", "Historical Tidbit:", "Expert's Musings:").
*   Be informative but not overly verbose for the initial analysis. Aim for a knowledgeable yet approachable tone.
*   After this initial analysis, the user may ask follow-up questions.
"""

def process_with_model(image_input, messages_for_template):
    """
    Processes the image and messages with the MLX model.
    Prepends a system prompt to the conversation.
    Yields GenerationResult objects from stream_generate.
    """
    app.logger.info(f"process_with_model called. Image type: {type(image_input)}")
    
    # Prepare messages: Prepend system prompt if it's the start of a new analysis
    # A simple heuristic: if messages_for_template contains only one user message,
    # it's likely the initial prompt for analysis.
    actual_messages_for_template = []
    if len(messages_for_template) == 1 and messages_for_template[0].get("role") == "user":
        actual_messages_for_template.append({"role": "system", "content": SYSTEM_PROMPT})
        actual_messages_for_template.extend(messages_for_template)
        app.logger.info("System prompt prepended for initial analysis.")
    else:
        # For follow-up messages, or if system prompt was already included by a more sophisticated frontend
        actual_messages_for_template = messages_for_template

    app.logger.info(f"Messages for template (with system prompt if added): {actual_messages_for_template}")

    if not model or not processor or not config:
        app.logger.error("Model, processor, or config not loaded.")
        yield {"error_detail": "Model not loaded. Please check server logs."}
        return

    images_for_model = []
    if isinstance(image_input, str): # URL
        images_for_model = [image_input]
    elif isinstance(image_input, Image.Image): # PIL Image
        images_for_model = [image_input]
    else:
        yield {"error_detail": "Invalid image input type. Must be a URL string or PIL Image."}
        return

    try:
        app.logger.info("Applying chat template...")
        formatted_prompt = apply_chat_template(
            processor, config, actual_messages_for_template, num_images=len(images_for_model)
        )
        app.logger.info(f"Formatted prompt: {formatted_prompt}")
        
        app.logger.info("Streaming generation with model...")
        # Use stream_generate which should yield tokens directly
        token_count = 0
        for token in stream_generate(model, processor, formatted_prompt, images_for_model):
            yield token
            token_count += 1
        
        app.logger.info(f"stream_generate finished. Total tokens yielded: {token_count}")
        if token_count == 0:
            app.logger.info("stream_generate yielded no tokens. Yielding one empty string for EOS.")
            yield "" # Ensure at least one signal goes to event_stream_generator if no tokens
            
    except Exception as e:
        app.logger.error(f"Error during model streaming: {e}", exc_info=True)
        yield {"error_detail": f"Model streaming error: {str(e)}"}

@app.route('/api/generate', methods=['POST'])
def handle_generate():
    if not model:
        # For SSE, errors should also be in the event stream if possible, or a non-200 initial response.
        # Here, a direct error response before streaming starts is okay.
        return jsonify({"error": "Model not available. Check server logs."}), 503

    # Changed 'prompt' to 'messages' which is a JSON string of the chat history
    if 'messages' not in request.form:
        return jsonify({"error": "Messages (chat history) are required"}), 400

    try:
        messages_str = request.form['messages']
        messages = json.loads(messages_str)
        if not isinstance(messages, list) or not all(isinstance(m, dict) and 'role' in m and 'content' in m for m in messages):
            return jsonify({"error": "Invalid 'messages' format. Expected a list of {'role': ..., 'content': ...} objects."}), 400
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON in 'messages' field."}), 400
    except Exception as e:
        return jsonify({"error": f"Error processing messages: {str(e)}"}), 400

    image_data_for_processing = None
    temp_image_path = None 

    # Image handling remains largely the same
    # It's required for every call in this VLM context
    if 'image_file' in request.files and request.files['image_file'].filename != '':
        file = request.files['image_file']
        filename = str(uuid.uuid4()) + "_" + file.filename 
        temp_image_path = os.path.join(UPLOAD_FOLDER, filename)
        try:
            file.save(temp_image_path)
            image_data_for_processing = Image.open(temp_image_path)
            image_data_for_processing.load() 
        except Exception as e:
            app.logger.error(f"Error processing uploaded image file: {e}", exc_info=True)
            return jsonify({"error": f"Error processing uploaded image: {str(e)}"}), 500
            
    elif 'image_url' in request.form and request.form['image_url'].strip():
        image_data_for_processing = request.form['image_url'].strip()
    else:
        # If chat history exists, we might not need a *new* image, but the current model needs one.
        # For simplicity, require image data on every call for now.
        return jsonify({"error": "Either image_file (upload) or image_url is required for VLM context"}), 400

    if image_data_for_processing is None:
         return jsonify({"error": "Image data could not be processed or was not provided"}), 400

    def event_stream_generator(current_image_data, current_messages):
        # current_temp_image_path needs to be accessible if image_data is a PIL.Image
        # For URLs, current_image_data is the URL string.
        # For file uploads, current_image_data is a PIL.Image object.
        active_temp_path = temp_image_path if isinstance(current_image_data, Image.Image) else None
        
        try:
            for content_part in process_with_model(current_image_data, current_messages):
                print(content_part.text)
                #content_part is <class 'mlx_vlm.utils.GenerationResult'>, get tokens from content_part


                if isinstance(content_part, dict) and 'error_detail' in content_part:
                    error_payload = json.dumps({"error": content_part['error_detail']})
                    yield f"data: {error_payload}\n\n"
                    return 
                elif isinstance(content_part, mlx_vlm.utils.GenerationResult):
                    payload = json.dumps({"token": content_part.text})
                    yield f"data: {payload}\n\n"
                # time.sleep(0.01) # Optional small delay for SSE
            yield f"data: {json.dumps({'event': 'eos'})}\n\n" # End of stream signal
        except Exception as e:
            app.logger.error(f"Critical error in event_stream_generator: {e}", exc_info=True)
            error_payload = json.dumps({"error": f"Stream generation critical error: {str(e)}"})
            yield f"data: {error_payload}\n\n"
        finally:
            # Clean up PIL image object if it was opened
            if isinstance(current_image_data, Image.Image):
                 current_image_data.close()
            # Clean up temporary file if it was created
            if active_temp_path and os.path.exists(active_temp_path):
                try:
                    os.remove(active_temp_path)
                except Exception as e_remove:
                    app.logger.error(f"Error removing temp file {active_temp_path}: {e_remove}")

    # Pass the actual image data and messages to the generator
    return Response(stream_with_context(event_stream_generator(image_data_for_processing, messages)), mimetype='text/event-stream')

if __name__ == '__main__':
    # Note: For production, use a proper WSGI server like Gunicorn or Waitress.
    # Debug mode should be False in production.
    app.logger.setLevel(logging.INFO) # Set log level to INFO
    app.run(host='0.0.0.0', port=5001, debug=False) 