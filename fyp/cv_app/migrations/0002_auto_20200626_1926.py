# Generated by Django 3.0.5 on 2020-06-26 14:26

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('cv_app', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='image',
            name='name',
            field=models.CharField(max_length=30),
        ),
    ]
