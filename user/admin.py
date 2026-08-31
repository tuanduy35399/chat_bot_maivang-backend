from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
# Register your models here.
from .models import User

@admin.register(User)
class AccountAdmin(UserAdmin):
    list_display= ('id', 'username', 'name', 'email', 'is_staff', 'is_active')
    list_filter=('is_staff', 'is_active')
    search_fields= ('username', 'name', 'email')
    fieldsets= UserAdmin.fieldsets + (
        ('Thong tin bo sung', {'fields':('name',)}),
    )