import os
import requests
from flask import Flask, request, render_template
from transformers import AutoProcessor, BlipForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer
from PIL import Image
from groq import Groq

app = Flask(__name__)

# Load the BLIP model and processor
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Initialize Groq client
client = Groq(api_key='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')

# Load conversational model and tokenizer (e.g., DialoGPT)
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
chatbot_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

recent_chats = []

@app.route('/')
def main():
    return render_template('main.html')

@app.route('/index', methods=['GET', 'POST'])
def index():
    facts = None
    topic = None
    error = None

    if request.method == 'POST':
        user_query = request.form.get('user_query', None)

        # Check if an image file was uploaded
        if 'image' in request.files and request.files['image'].filename != '':
            file = request.files['image']
            try:
                image = Image.open(file.stream)
                inputs = processor(images=image, return_tensors="pt")
                outputs = model.generate(**inputs)
                generated_caption = processor.decode(outputs[0], skip_special_tokens=True)

                # Update role prompt based on the caption
                role_prompt = (
                    "You are an intelligent design assistant specialized in PCB defect detection and hardware design. "
                    "You should respond to queries related to PCB defects and hardware design. "
                    "Analyze the detected issue below and provide innovative solutions and possible defect resolutions:\n"
                    "Detected Issue: " + generated_caption + "\n"
                    "Possible Solutions:"
                )

                # Generate response using Groq API
                chat_completion = client.chat.completions.create(
                    messages=[{"role": "user", "content": role_prompt}],
                    model="llama3-70b-8192",
                )
                generated_text = chat_completion.choices[0].message.content

                facts = f"Detected Issues: {generated_caption}. Analysis: {generated_text}"
                recent_chats.append({'topic': 'PCB Defects', 'facts': facts})

            except Exception as e:
                error = f"An error occurred while processing the image: {str(e)}"

        # Handle text inquiries using the conversational model
        elif user_query:
            # Generate a response using DialoGPT
            input_ids = tokenizer.encode(user_query + tokenizer.eos_token, return_tensors='pt')
            chat_history_ids = None

            # Generate response
            if recent_chats:
                # Concatenate previous conversations
                chat_history_ids = recent_chats[-1]['facts'] + input_ids

            response_ids = chatbot_model.generate(
                chat_history_ids,
                max_length=1000,
                pad_token_id=tokenizer.eos_token_id,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                do_sample=True
            )
            response_text = tokenizer.decode(response_ids[:, chat_history_ids.shape[-1]:][0], skip_special_tokens=True)

            facts = str(response_text)
            recent_chats.append({'topic': 'User Inquiry', 'facts': facts})

        else:
            error = "Please upload an PCB BOARD image"

    return render_template('index.html', facts=facts, topic=topic, error=error, recent_chats=recent_chats)

if __name__ == '__main__':
    app.run(debug=True)
