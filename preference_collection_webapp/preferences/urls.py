import os
from django.conf.urls import url
from django.urls import path
from django.views.static import serve

from preferences import views
from pbrlwebapp import settings

urlpatterns = [
    path('', views.QueryListView.as_view(), name='index'),
    path('next', views.next_query, name='next'),
    path('next/', views.next_query, name='next'),
    path('<uuid:query_id>', views.query, name='query'),
    path('<uuid:query_id>/', views.query, name='query'),
    url(r'^(?P<path>.*)$', serve,
        {'document_root': settings.BASE_DIR.parent / 'videofiles'})
]
debug = 1+1

