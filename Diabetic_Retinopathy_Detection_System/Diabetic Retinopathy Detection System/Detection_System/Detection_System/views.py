import os
from os.path import join, dirname, realpath
from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage

from .models import DetectionModel


def index(request):
    return render(request, 'home.html')


def process_image(request):
    if request.method == 'POST' and request.FILES['upload']:
        upload = request.FILES['upload']
        fss = FileSystemStorage()
        file = fss.save(upload.name, upload)
        file_url = fss.url(file)
        result = DetectionModel().process_image(file_url)
        return render(request, 'home.html', {'result': result})




