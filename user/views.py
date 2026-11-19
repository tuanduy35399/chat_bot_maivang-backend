import logging
import secrets
from datetime import timedelta

from django.conf import settings
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
from django.core.mail import send_mail
from django.db import transaction
from django.utils import timezone
from django.utils.crypto import constant_time_compare, salted_hmac
from rest_framework import generics, permissions, status
from rest_framework.throttling import SimpleRateThrottle
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import User, PasswordResetOTP
from .serializers import (UserSerialzer, RegisterSerializer,
                          PasswordResetRequestSerializer, PasswordResetConfirmSerializer)

logger = logging.getLogger(__name__)


class ResetThrottle(SimpleRateThrottle):
    scope = 'password_reset'
    rate = '20/hour'

    def get_cache_key(self, request, view):
        return self.cache_format % {'scope': self.scope, 'ident': self.get_ident(request)}


def otp_hash(user_id, code):
    return salted_hmac('user.password_reset', f'{user_id}:{code}', secret=settings.SECRET_KEY, algorithm='sha256').hexdigest()


def find_reset_user(email):
    # Avoid resetting the wrong account if legacy records differ only by case.
    users = list(User.objects.filter(email__iexact=email.strip(), is_active=True)[:2])
    return users[0] if len(users) == 1 else None


class PasswordResetRequestView(APIView):
    permission_classes = [permissions.AllowAny]
    authentication_classes = []
    throttle_classes = [ResetThrottle]
    serializer_class = PasswordResetRequestSerializer

    def post(self, request):
        serializer = self.serializer_class(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = find_reset_user(serializer.validated_data['email'])
        if user:
            try:
                with transaction.atomic():
                    user = User.objects.select_for_update().get(pk=user.pk)
                    previous = PasswordResetOTP.objects.filter(user=user).first()
                    now = timezone.now()
                    if not previous or previous.created_at <= now - timedelta(seconds=60):
                        code = f'{secrets.randbelow(1000000):06d}'
                        PasswordResetOTP.objects.update_or_create(user=user, defaults={
                            'code_hash': otp_hash(user.pk, code), 'created_at': now, 'attempts': 0,
                        })
                        send_mail('Mã khôi phục mật khẩu MaiCare',
                                  f'Mã OTP của bạn là {code}. Mã có hiệu lực trong 10 phút. Không chia sẻ mã này với người khác.',
                                  settings.DEFAULT_FROM_EMAIL, [user.email], fail_silently=False)
            except Exception:
                logger.exception('Password reset email delivery failed')
                return Response({'detail': 'Không thể gửi email lúc này. Vui lòng thử lại sau.'}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
        return Response({'detail': 'Nếu email đã được đăng ký, mã OTP sẽ được gửi đến email đó.'})


class PasswordResetConfirmView(APIView):
    permission_classes = [permissions.AllowAny]
    authentication_classes = []
    throttle_classes = [ResetThrottle]
    serializer_class = PasswordResetConfirmSerializer

    def post(self, request):
        serializer = self.serializer_class(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        user = find_reset_user(data['email'])
        invalid = Response({'otp': ['Mã OTP không hợp lệ hoặc đã hết hạn.']}, status=status.HTTP_400_BAD_REQUEST)
        if not user:
            return invalid
        with transaction.atomic():
            user = User.objects.select_for_update().get(pk=user.pk)
            otp = PasswordResetOTP.objects.filter(user=user).first()
            if not otp or otp.created_at <= timezone.now() - timedelta(minutes=10) or otp.attempts >= 5:
                return invalid
            if not constant_time_compare(otp.code_hash, otp_hash(user.pk, data['otp'])):
                otp.attempts += 1
                otp.save(update_fields=['attempts'])
                return invalid
            try:
                validate_password(data['new_password'], user)
            except ValidationError as exc:
                return Response({'new_password': list(exc.messages)}, status=status.HTTP_400_BAD_REQUEST)
            user.set_password(data['new_password'])
            user.save(update_fields=['password'])
            otp.delete()
        return Response({'detail': 'Mật khẩu đã được cập nhật.'})


class RegisterView(generics.CreateAPIView):
    """
    POST /api/v1/user/register/
    """
    
    queryset= User.objects.all()
    serializer_class= RegisterSerializer
    permission_classes= [permissions.AllowAny]
    
class MeView(generics.RetrieveUpdateDestroyAPIView):
    """ 
    GET, PATCH, DELETE /api/v1/user/me/
    """
    serializer_class=UserSerialzer
    permission_classes= [permissions.IsAuthenticated]
    
    def get_object(self): #Khong dung id de tim user de tranh lo id user khi truyen url qua internet
        return self.request.user #DRF se tu parse userid tu token
