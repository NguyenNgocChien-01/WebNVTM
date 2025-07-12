
from django.contrib import admin
from django.urls import path

from home.views import *

urlpatterns = [
    path('',trangchu, name='trangchu'),
    path('run-real/', ocr_and_predict_view, name='run-real'),
    path('test/',run_test, name='test'),
    path('real/',run_real, name='real'),
    path('run-test/',run_by_index_view, name='run-test'),

]
