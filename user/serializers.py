from django.contrib.auth.password_validation import validate_password
from rest_framework import serializers

from .models import User

class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only= True, validators =[validate_password])
    
    class Meta:
        model= User
        fields = ('id', 'username','name', 'email','password')
    def create(self, validated_data):
        user=User(
            username= validated_data['username'],
            name= validated_data['name'],
            email= validated_data['email'],
        )
        user.set_password(validated_data['password']) #dat mk
        user.save()
        return user
    
class UserSerialzer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ('id', 'username', 'name', 'email', 'is_staff', 'date_joined')
        read_only_fields = ('id', 'username', 'is_staff', 'data_joined')


class PasswordResetRequestSerializer(serializers.Serializer):
    email = serializers.EmailField()


class PasswordResetConfirmSerializer(PasswordResetRequestSerializer):
    otp = serializers.RegexField(r'^\d{6}$')
    new_password = serializers.CharField(write_only=True, trim_whitespace=False)
