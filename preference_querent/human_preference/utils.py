import os
import sys
import django


def ensure_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir


def prepare_django_connection():
    sys.path.append(os.path.abspath('./preference_collection_webapp'))
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pbrlwebapp.settings')
    django.setup()
