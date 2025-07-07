from django.urls import path
from .import views
urlpatterns = [
    path('', views.weather_view, name='weather_view'),
    path('arima/', views.arima_view, name='arima_view'),
    path('sarima/', views.sarima_view, name='sarima_view'),
    path('var/', views.var_view, name='var_view'),
    path('varima/', views.varima_view, name='varima_view'),
    path('compare/', views.compare_view, name='compare_view'),
    # Add more URL patterns as needed
]