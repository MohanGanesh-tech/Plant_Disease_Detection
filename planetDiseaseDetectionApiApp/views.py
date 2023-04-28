from django.shortcuts import render
from django.http import JsonResponse
import numpy as np
from tensorflow.keras.models import load_model
from django.views.decorators.csrf import csrf_exempt, ensure_csrf_cookie
from PIL import Image
# Create your views here.

@ensure_csrf_cookie
@csrf_exempt
def test(request):
    return JsonResponse({'message': 'Test Successfull'})

@ensure_csrf_cookie
@csrf_exempt
def predict(request):
    if request.method == 'POST':
        image_file = request.FILES['image']
        img = Image.open(image_file)
        img = img.resize((256, 256)) 
        image = np.array(img)
        try:
            model = load_model('C:/Users/ganesh/Desktop/Plant Disease Detection System/planetDiseaseDetection/static/machineLearningModels/plant_disease_detection_model.h5')
            class_names = np.load('C:/Users/ganesh/Desktop/Plant Disease Detection System/planetDiseaseDetection/static/machineLearningModels/class_names.npy')
            image = np.expand_dims(image, 0)
            predictions = model.predict(image)
            confidence = np.max(predictions[0])
            predicted_class = class_names[np.argmax(predictions[0])]
            print(predicted_class, confidence)
            return JsonResponse({'predicted class': predicted_class,'confidence': str(confidence)})

        except Exception as e: print(e)

    return JsonResponse({'message': 'Something is wrong try later'})
