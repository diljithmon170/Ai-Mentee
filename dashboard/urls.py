from get_std.views import home
from log_sig.views import log_sig
from dashboard.views import dashboard
from django.urls import path, include
from . import views
from django.shortcuts import render
from .views import quiz_view

urlpatterns = [
    path('', views.dashboard, name='dashboard'),  # Dashboard page
    path('level/<str:course_name>/', views.level, name='level'),  # Level page with course parameter
    path('quiz/<str:course>/', views.quiz_view, name='quiz'),  # Quiz page with course parameter
    # path('text/<str:level>/', views.text_view, name='text'),  # Text page with level parameter
    path('content/<str:course_name>/<str:level>/', views.content_view, name='content'),  # Content page with course and level parameters
    path('llama3/<str:course>/', views.generate_content, name='text'),
    path('tts/<str:course>/', views.generate_content, name='audio'),
    path('ttv/<str:course>/', views.generate_content, name='video'),
    

    path('text/<str:course_name>/<str:level>/<int:file_number>/', views.text_view, name='text'),
    path('audio/<str:course_name>/<str:level>/<int:file_number>/', views.audio_view, name='audio'),
    path('video/<str:course_name>/<str:level>/<int:file_number>/', views.video_view, name='video'),

    path('enroll/<str:course_name>/', views.enroll_course, name='enroll_course'),
    path('complete/<str:course_name>/', views.complete_course, name='complete_course'),
]
