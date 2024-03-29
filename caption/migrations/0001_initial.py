# Generated by Django 2.2 on 2022-04-01 15:22

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Image',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('caption_given', models.CharField(max_length=512)),
                ('caption_generated', models.CharField(max_length=512)),
                ('image', models.ImageField(upload_to='uploaded_images')),
            ],
        ),
    ]
