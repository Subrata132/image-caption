from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import Image


def test_view(request):
    return render(request, 'home.html')


class ImageView(APIView):
    def post(self, request):
        image = request.FILES['image']
        new_image = Image.objects.create(image=image)
        return Response({}, status=status.HTTP_201_CREATED)