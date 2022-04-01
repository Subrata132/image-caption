from django.db import models


class Image(models.Model):
    caption_given = models.CharField(max_length=512)
    caption_generated = models.CharField(max_length=512)
    image = models.ImageField(upload_to='uploaded_images')

    def save(self, *args, **kwargs):
        ext = self.image.name.split('.')[-1]
        try:
            current_id = Image.objects.last().id
        except:
            current_id = 0
        self.image.name = f'image_{current_id+1}.{ext}'
        super(Image, self).save(*args, **kwargs)

