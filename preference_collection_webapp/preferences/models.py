from django.db import models


class Preference(models.Model):

    uuid = models.UUIDField('query_uuid', db_index=True)

    created_timestamp = models.DateTimeField(
        'created at', auto_now_add=True, db_index=True)
    priority = models.IntegerField('priority', db_index=True, default=0)

    label = models.DecimalField('preference label', max_digits=2,
                                decimal_places=1, db_index=True, blank=True, null=True)

    def __str__(self) -> str:
        return str(self.id)

    @property
    def video_file_path_left(self) -> str:
        return '{}-left.webm'.format(self.uuid)

    @property
    def video_file_path_right(self) -> str:
        return '{}-right.webm'.format(self.uuid)
