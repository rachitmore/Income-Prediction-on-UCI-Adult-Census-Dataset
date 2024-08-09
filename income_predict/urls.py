from django.urls import path
from . import views

app_name = 'income_predict'

urlpatterns = [
    path('', views.home, name='home'),

]
