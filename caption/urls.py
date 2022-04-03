from django.urls import path
from .views import *

urlpatterns = [
    path('', test_view, name='test_view'),
    path('upload', ImageView.as_view(), name='upload'),
]
