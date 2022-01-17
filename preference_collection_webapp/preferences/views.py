from django.views.generic.list import ListView
from django.http.response import Http404
from django.shortcuts import get_object_or_404, redirect, render

from .models import Preference

class QueryListView(ListView):

    model = Preference
    paginated_by = 100
    ordering = 'created_timestamp'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        return context

def index(request):
    
    return render(request, 'preferences/index.html')


def next_query(request):
    top_query = Preference.objects.filter(label__isnull=True).order_by('priority').first()
    if top_query is not None:
        return redirect('query', query_id=top_query.uuid)
    else:
        return redirect('index')


def query(request, query_id):

    query = get_object_or_404(Preference, uuid=query_id)
    if request.method == 'POST' and (label := request.POST['action']) is not None:
        if label == 'Left':
            query.label = 1
        elif label =='Right':
            query.label = 0
        elif label =='Indifferent':
            query.label = .5
        elif label == 'Skip':
            query.label = -1
        query.full_clean()
        query.save()
        return redirect('next')

    context = {
        'query': query,
        'video_url_left': query.video_file_path_left,
        'video_url_right': query.video_file_path_right
    }
    return render(request, 'preferences/query.html', context)
