from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from ultralytics import YOLO
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

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

def download_images(url, folder_name, max_images=8):
    """Downloads high-quality images from a given URL.

    Args:
        url: The URL of the webpage containing the images.
        folder_name: The name of the folder to save the images.
        max_images: The maximum number of images to download.
    """

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    response = requests.get(url)
    response.raise_for_status()  # Check for HTTP errors

    soup = BeautifulSoup(response.content, 'html.parser')

    images = soup.find_all('img')

    count = 0
    for i, img in enumerate(images):
        if count >= max_images:
            break

        img_url = img.get('src')

        if not img_url:
            continue

        # Ensure the image URL is absolute
        if not img_url.startswith('http'):
            img_url = urljoin(url, img_url)

        try:
            img_data = requests.get(img_url).content
            with open(f'{folder_name}/image_{count + 1}.jpg', 'wb') as handler:
                handler.write(img_data)
            print(f'Downloaded: image_{count + 1}.jpg')
            count += 1

        except Exception as e:
            print(f'Error downloading {img_url}: {e}')


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
    class_folder = os.path.join(app.config['RECOMMENDATIONS_FOLDER'], class_name)
    if not os.path.exists(class_folder):
        os.makedirs(class_folder)

    # Fetch high-quality fashion recommendation images using Unsplash
    query = f"{class_name} fashion"
    unsplash_url = f"https://unsplash.com/s/photos/{query.replace(' ', '-')}"
    download_images(unsplash_url, folder_name=class_folder, max_images=8)

    # List downloaded images
    local_images = [f for f in os.listdir(class_folder) if f.startswith("image")]

    return render_template('recommendations.html', class_name=class_name, images=local_images)


@app.route('/recommendations_img/<class_name>/<filename>')
def recommendations_img(class_name, filename):
    class_folder = os.path.join(app.config['RECOMMENDATIONS_FOLDER'], class_name)
    return send_from_directory(class_folder, filename)


if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    if not os.path.exists(app.config['RECOMMENDATIONS_FOLDER']):
        os.makedirs(app.config['RECOMMENDATIONS_FOLDER'])
    app.run(debug=True)
