from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from ultralytics import YOLO
import os
import requests
from bs4 import BeautifulSoup
import random

# Initialize the Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RECOMMENDATIONS_FOLDER'] = 'recommendations'

# List of class names corresponding to indices
class_names = [
    "dress", "t-shirt", "pants", "outwear", "shoes",
    "shirt", "hat", "shorts", "longsleeve", "skirt"
]

# Load a pretrained YOLOv8n-cls Classify model
model = YOLO("best.pt")


def get_high_res_image_url(thumb_url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    response = requests.get(thumb_url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    high_res_img = soup.find('img', class_='n3VNCb')
    if high_res_img and high_res_img.has_attr('src'):
        src = high_res_img['src']
        if "w=" in src:
            # Increase resolution by modifying the URL
            parts = src.split('&')
            for i in range(len(parts)):
                if parts[i].startswith('w='):
                    parts[i] = 'w=1920'
                if parts[i].startswith('h='):
                    parts[i] = 'h=1080'
            return '&'.join(parts)
        return src
    return thumb_url


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Run inference on the uploaded image
        results = model(file_path)  # results list
        top1_class_name = ""

        for r in results:
            if hasattr(r, 'probs'):
                top1 = r.probs.top1
                top1_class_name = class_names[top1]

        return redirect(url_for('recommendations', class_name=top1_class_name))


@app.route('/recommendations/<class_name>')
def recommendations(class_name):
    if not os.path.exists(app.config['RECOMMENDATIONS_FOLDER']):
        os.makedirs(app.config['RECOMMENDATIONS_FOLDER'])

    # Fetch high-quality fashion recommendation images using web scraping
    query = f"{class_name} fashion"
    search_url = f"https://www.google.com/search?q={query}&tbm=isch"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    images = []
    for img in soup.find_all('img', limit=50):  # Increase limit to get more high-res options
        thumb_url = img.get('data-src') or img.get('src')
        if thumb_url and thumb_url.startswith('http'):
            high_res_url = get_high_res_image_url(thumb_url)
            images.append(high_res_url)

    # Randomly select 6 images if enough high-res images are found
    selected_images = random.sample(images, min(6, len(images)))

    # Download and save images
    local_images = []
    for i, img_url in enumerate(selected_images):
        save_path = os.path.join(app.config['RECOMMENDATIONS_FOLDER'], f"{class_name}_{i}.jpg")
        response = requests.get(img_url)
        with open(save_path, 'wb') as f:
            f.write(response.content)
        local_images.append(f"{class_name}_{i}.jpg")

    return render_template('recommendations.html', class_name=class_name, images=local_images)


@app.route('/recommendations_img/<filename>')
def recommendations_img(filename):
    return send_from_directory(app.config['RECOMMENDATIONS_FOLDER'], filename)


if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    if not os.path.exists(app.config['RECOMMENDATIONS_FOLDER']):
        os.makedirs(app.config['RECOMMENDATIONS_FOLDER'])
    app.run(debug=True)
