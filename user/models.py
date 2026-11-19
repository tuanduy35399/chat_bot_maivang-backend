from django.db import models
from django.contrib.auth.models import AbstractUser
# Create your models here.
class User(AbstractUser): #ke thua tu User co san cua Django
    name= models.CharField(max_length=200, blank=False, null=False)
    email= models.EmailField(unique=True)
    
    USERNAME_FIELD= 'username'
    REQUIRED_FIELDS = ['email']
    
    def __str__(self):
        return self.username


class PasswordResetOTP(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='password_reset_otp')
    code_hash = models.CharField(max_length=64)
    created_at = models.DateTimeField()
    attempts = models.PositiveSmallIntegerField(default=0)
