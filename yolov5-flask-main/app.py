import io
import os
import json
from flask_ngrok import run_with_ngrok
from PIL import Image

import torch
from flask import Flask, jsonify, url_for, render_template, request, redirect

app = Flask(__name__)
run_with_ngrok(app)

RESULT_FOLDER = os.path.join('yolov5-flask-main/static')
app.config['RESULT_FOLDER'] = RESULT_FOLDER

model = torch.hub.load('ultralytics/yolov5', 'custom', path='model/best.pt')  # default
model.eval()

def get_prediction(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    imgs = [img]  # batched list of images

# Inference
    results = model(imgs, size=640)  # includes NMS
    return results

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return

        img_bytes = file.read()
        results = get_prediction(img_bytes)
        results.save('yolov5-flask-main/static')  # save as results1.jpg, results2.jpg... etc.
        os.rename("yolov5-flask-main/static/image0.jpg", "yolov5-flask-main/static/results0.jpg")

        full_filename = os.path.join(app.config['RESULT_FOLDER'], 'results0.jpg')
        return redirect('static/results0.jpg')
    return render_template('index.html')

if __name__ == '__main__':
    app.run()
