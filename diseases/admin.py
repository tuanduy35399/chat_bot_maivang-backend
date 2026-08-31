from django.contrib import admin
from .models import DiseaseDetail
# Register your models here.
@admin.register(DiseaseDetail)
class DiseaseDetailAdmin(admin.ModelAdmin):
    list_display=('id', 'name', 'prompt', 'img_ex')
    search_fields= ('name', 'info', )