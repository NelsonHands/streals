from flask import Flask, request, jsonify, render_template
import base64
import requests
import os
from dotenv import load_dotenv
import pandas as pd
import json

# Load environment variables from .env file
load_dotenv()

# OpenAI API Key
api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

# Function to encode the image
def encode_image(image):
    return base64.b64encode(image.read()).decode('utf-8')

# Function to send request to OpenAI API
def analyze_image(base64_image, data_points):
    # Create a custom GPT prompt based on user-inputted data points
    prompt = f"Please extract the following data points and return them in a flat JSON structure with only one level. Each data point should have exactly one value, and there should be no nested objects or arrays in the output. Use the following structure: 'data_point_1': 'value_for_data_point_1', 'data_point_2': 'value_for_data_point_2',... Instructions: 1. The data points to be extracted are: {data_points}. 2. Each data point should correspond to a single key in the JSON, with only one value. 3. The JSON structure should be flat, meaning no nested objects or arrays are allowed. 4. Each requested detail should appear exactly once in the output. 5. Ensure that all values are directly under the top-level keys."

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }

    # Send request to OpenAI API
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    
    # Print the full response from OpenAI API
    print("ChatGPT Response:")
    print(response.json())  # Print the full JSON response for debugging purposes
    
    return response.json()

@app.route('/')
def index():
    return render_template('index.html')

def flatten_json(y):
    out = {}

    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '_')
        else:
            out[name[:-1]] = x

    flatten(y)
    return out

@app.route('/upload', methods=['POST'])
def upload():
    if 'images' not in request.files:
        return jsonify({'error': 'No images uploaded'}), 400

    files = request.files.getlist('images')
    data_points = request.form['dataPoints']

    all_responses = []

    for image in files:
        base64_image = encode_image(image)
        response = analyze_image(base64_image, data_points)

        output_json = response['choices'][0]['message']['content']
        output_json = output_json.replace("```json", "").replace("```", "").strip()

        try:
            output_json = output_json.replace("\n", "")
            data = json.loads(output_json)

            if isinstance(data, dict):
                data = [data]

            # Flatten each response
            for item in data:
                flattened_item = flatten_json(item)
                all_responses.append(flattened_item)
                
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error: {e}")
            return jsonify({'error': f'Failed to parse JSON: {e}'}), 400

    df = pd.DataFrame(all_responses)

    return df.to_json(orient='records')

if __name__ == '__main__':
    app.run(debug=True)
