from django.http.response import Http404
from django.shortcuts import get_object_or_404, redirect, render

from .models import Preference


def index(request):
    latest_queries = Preference.objects.order_by('created_timestamp')[:5]
    context = {
        'latest_queries': latest_queries,
    }
    return render(request, 'preferences/index.html', context)


def next_query(request):
    top_query = Preference.objects.filter(label__isnull=True).first()
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
        else:
            #TODO implement proper skip functionality
            pass
        query.full_clean()
        query.save()
        return redirect('next')

    context = {
        'query': query,
        'video_url_left': query.video_file_path_left,
        'video_url_right': query.video_file_path_right
    }
    return render(request, 'preferences/query.html', context)
