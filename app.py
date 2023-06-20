import io
import os
import json
# from flask_ngrok import run_with_ngrok
from PIL import Image
import shutil
import torch
from flask import Flask, jsonify, url_for, render_template, request, redirect

app = Flask(__name__)
# run_with_ngrok(app)

RESULT_FOLDER = os.path.join(os.getcwd() + '/static')
app.config['RESULT_FOLDER'] = RESULT_FOLDER

model = torch.hub.load('ultralytics/yolov5', 'custom', path=os.getcwd() + '/model/best.pt')  # default
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
        if os.path.exists(os.path.join(os.getcwd(), 'static')):
        	shutil.rmtree(os.path.join(os.getcwd(), 'static'))
        # results.save(save_dir="/home/tim/PycharmProjects/DataScience/computer_vision/mm-cctv-gun-detection/static/")
        # shutil.rmtree(os.path.join(os.getcwd(), "runs"))
        results.save(save_dir=os.path.join(os.getcwd(), 'static'))  # save as results1.jpg, results2.jpg... etc.
        os.rename((os.path.join(os.getcwd() , "static/image0.jpg")), (os.path.join(os.getcwd() , "static/results0.jpg")))

        # full_filename = os.path.join(app.config['RESULT_FOLDER'], 'results0.jpg')
        # return redirect('static/results0.jpg')
        print(str(results.pandas().xyxy[0]["name"]).split()[1])
        
    if str(results.pandas().xyxy[0]["name"]).split()[1] == "pistol":
        
    	return jsonify({"pred": "pred_made"})

if __name__ == '__main__':
    app.run()
