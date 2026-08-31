from django.db import models

# Create your models here.
class DiseaseDetail(models.Model):
    name = models.CharField(max_length=200, verbose_name='Ten benh')
    infor= models.TextField(verbose_name='Thong tin ve benh')
    prompt = models.TextField(
        blank= True,
        verbose_name='prompt',
        help_text="Thong tin mo ta benh de dua len LLM, cai nay admin dat"
    )
    img_ex= models.ImageField(
        upload_to='diseases/', blank=True, null=True, verbose_name='Hinh minh hoa cho benh'
    )
    class Meta:#khai bao metadata cho model (hien thi tren trang admin)
        verbose_name= 'Loai benh'  #hien thi so it cua model trong trang admin
        verbose_name_plural='So tay benh tren cay mai' #hien thi so nhieu cua model trong trang admin
        ordering=['name'] #sap xep theo ten khi goi DiseaseDetail.objects.all() ma khong can .order_by('name')
        
    def __str__(self):
        return self.name