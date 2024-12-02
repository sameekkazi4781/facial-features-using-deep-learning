from django.db import models


class Image(models.Model):
    name = models.CharField(max_length=30)
    image = models.ImageField()