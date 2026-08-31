from rest_framework import serializers

from .models import DiseaseDetail

class DiseaseDetailSerializer(serializers.ModelSerializer):
    class Meta:
        model=DiseaseDetail
        fields = ('id','name', 'infor', 'prompt', 'img_ex')