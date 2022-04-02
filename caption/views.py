from django.shortcuts import render
from django.conf import settings
from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import Image
from .api import get_image


def test_view(request):
    return render(request, 'home.html')


class ImageView(APIView):
    def post(self, request):
        image = request.FILES['image']
        new_image = Image.objects.create(image=image)
        caption = settings.CAP_OBJ.caption_generator(new_image.image.name.split('/')[-1])
        return Response(caption, status=status.HTTP_201_CREATED)


def image_web(request):
    print('python')
    get_image(request)
    return JsonResponse({'a': 'successful'})