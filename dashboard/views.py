from django.shortcuts import render, redirect, get_object_or_404
from log_sig.models import CustomUser, Course  # Import CustomUser from log_sig.models
from django.contrib.auth.decorators import login_required
from .forms import QuizForm
import os

@login_required
def dashboard(request):
    enrolled_courses = request.user.enrolled_courses.all()  # Assuming a ManyToMany relationship
    return render(request, 'dashboard.html', {'enrolled_courses': enrolled_courses})

@login_required
def level(request, course_name):
    """Render the level page for a specific course."""
    course_display_name = course_name  # Default to the original course name
    if course_name.lower() == "ai":
        course_display_name = "Artificial Intelligence"
    if course_name.lower() == "dbms":
        course_display_name = "Database Management System"
    if course_name.lower() == "ml":
        course_display_name = "Machine Learning"
    if course_name.lower() == "python":
        course_display_name = "Python"
    if course_name.lower() == "java":
        course_display_name = "Java"
    return render(request, 'level.html', {'course': course_name, 'course_display_name': course_display_name})

@login_required
def quiz_view(request, course):
    """Render the quiz page for a specific course."""
    course_display_name = course  # Default to the original course name
    if course.lower() == "ai":
        course_display_name = "Artificial Intelligence"
    if course.lower() == "dbms":
        course_display_name = "Database Management System"
    if course.lower() == "ml":
        course_display_name = "Machine Learning"
    if course.lower() == "python":
        course_display_name = "Python"
    if course.lower() == "java":
        course_display_name = "Java"
        
    form = QuizForm(course=course)
    if request.method == "POST":
        form = QuizForm(course=course, data=request.POST)
        if form.is_valid():
            score = form.check_answers()
            category = "Beginner" if score <= 5 else "Intermediate" if score <= 8 else "Expert"
            return render(request, 'quiz.html', {
                'form': form,
                'score': score,
                'category': category,
                'course': course,
                'course_display_name': course_display_name
            })
    return render(request, 'quiz.html', {
        'form': form,
        'course': course,
        'course_display_name': course_display_name
    })

@login_required
def text_view(request, course_name, level, file_number=1):
    """Render the text content page with navigation for files."""
    # Define the base directory for text files
    base_dir = os.path.join('dashboard', 'templates', 'interfaces', 'text_files')

    # Construct the file name based on the level and file number
    file_name = f"{level}{file_number}.txt"
    file_path = os.path.join(base_dir, file_name)

    # Read the content of the file
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        error = None
    except FileNotFoundError:
        content = ""
        error = f"The file {file_name} could not be found."

    # Determine if "Next" and "Previous" buttons should be enabled
    next_file = os.path.exists(os.path.join(base_dir, f"{level}{file_number + 1}.txt"))
    prev_file = file_number > 1 and os.path.exists(os.path.join(base_dir, f"{level}{file_number - 1}.txt"))

    # Render the text.html template with the course name, level, and file content
    return render(request, 'interfaces/text.html', {
        'course_name': course_name.title(),
        'level': level.title(),
        'content': content,
        'error': error,
        'file_number': file_number,
        'next_file': next_file,
        'prev_file': prev_file,
    })

@login_required
def content_view(request, level):
    """Render the content page based on the selected level and course."""
    # Retrieve the course name dynamically based on the level
    course_name = level.title()  # Assuming the level corresponds to the course name

    # Pass both the course name and level to the template
    return render(request, 'content.html', {
        'level': level,
        'course_name': course_name
    })



@login_required
def video_view(request, course_name, level, file_number=1):
    """Render the video content page with navigation for files."""
    base_dir = os.path.join('dashboard', 'static', 'videos')
    file_name = f"{level}{file_number}.mp4"
    file_path = os.path.join(base_dir, file_name)

    # Check if the file exists
    if not os.path.exists(file_path):
        error = f"The video file {file_name} could not be found."
        video_url = None
    else:
        error = None
        video_url = f"videos/{file_name}"

    # Determine if "Next" and "Previous" buttons should be enabled
    next_file = os.path.exists(os.path.join(base_dir, f"{level}{file_number + 1}.mp4"))
    prev_file = file_number > 1 and os.path.exists(os.path.join(base_dir, f"{level}{file_number - 1}.mp4"))

    return render(request, 'interfaces/video.html', {
        'course_name': course_name.title(),
        'level': level.title(),
        'video_url': video_url,
        'error': error,
        'file_number': file_number,
        'next_file': next_file,
        'prev_file': prev_file,
    })

@login_required
def audio_view(request, course_name, level, file_number=1):
    """Render the audio content page with navigation for files."""
    base_dir = os.path.join('dashboard', 'static', 'audio')
    file_name = f"{level}{file_number}.mp3"
    file_path = os.path.join(base_dir, file_name)

    # Check if the file exists
    if not os.path.exists(file_path):
        error = f"The audio file {file_name} could not be found."
        audio_url = None
    else:
        error = None
        audio_url = f"audio/{file_name}"

    # Determine if "Next" and "Previous" buttons should be enabled
    next_file = os.path.exists(os.path.join(base_dir, f"{level}{file_number + 1}.mp3"))
    prev_file = file_number > 1 and os.path.exists(os.path.join(base_dir, f"{level}{file_number - 1}.mp3"))

    return render(request, 'interfaces/audio.html', {
        'course_name': course_name.title(),
        'level': level.title(),
        'audio_url': audio_url,
        'error': error,
        'file_number': file_number,
        'next_file': next_file,
        'prev_file': prev_file,
    })



@login_required
def enroll_course(request, course_name):
    """Enroll the user in the selected course and redirect to level.html."""
    # Get the course object
    course = get_object_or_404(Course, name=course_name)

    # Add the course to the user's enrolled courses
    user = request.user
    user.enrolled_courses.add(course)

    # Redirect to the level.html page for the enrolled course
    return redirect('level', course_name=course_name)