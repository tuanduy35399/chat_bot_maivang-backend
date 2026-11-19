from datetime import timedelta
from unittest.mock import patch

from django.core import mail
from django.core.cache import cache
from django.test import override_settings
from django.utils import timezone
from rest_framework.test import APITestCase

from .models import PasswordResetOTP, User


@override_settings(EMAIL_BACKEND='django.core.mail.backends.locmem.EmailBackend', DEFAULT_FROM_EMAIL='test@example.com')
class PasswordResetTests(APITestCase):
    def setUp(self):
        cache.clear()
        self.user = User.objects.create_user(username='maicare', email='garden@gmail.com', password='Old-password-123', name='Garden')
        self.request_url = '/api/v1/user/password-reset/request/'
        self.confirm_url = '/api/v1/user/password-reset/confirm/'

    def request_code(self):
        with patch('user.views.secrets.randbelow', return_value=123):
            return self.client.post(self.request_url, {'email': 'GARDEN@gmail.com'}, format='json')

    def confirm(self, otp='000123', password='New-secure-456'):
        return self.client.post(self.confirm_url, {'email': 'garden@gmail.com', 'otp': otp, 'new_password': password}, format='json')

    def test_reset_and_single_use(self):
        self.assertEqual(self.request_code().status_code, 200)
        self.assertIn('000123', mail.outbox[0].body)
        self.assertNotIn('000123', PasswordResetOTP.objects.get(user=self.user).code_hash)
        self.assertEqual(self.confirm().status_code, 200)
        self.user.refresh_from_db()
        self.assertTrue(self.user.check_password('New-secure-456'))
        self.assertEqual(self.confirm().status_code, 400)
        self.assertEqual(self.client.post('/api/v1/user/login/', {'username': 'maicare', 'password': 'New-secure-456'}).status_code, 200)

    def test_unknown_email_and_resend_cooldown(self):
        self.assertEqual(self.client.post(self.request_url, {'email': 'unknown@gmail.com'}, format='json').status_code, 200)
        self.request_code()
        self.request_code()
        self.assertEqual(len(mail.outbox), 1)
        PasswordResetOTP.objects.filter(user=self.user).update(created_at=timezone.now() - timedelta(seconds=61))
        self.request_code()
        self.assertEqual(len(mail.outbox), 2)

    def test_expiry_attempt_limit_and_password_validation(self):
        self.request_code()
        self.assertEqual(self.confirm(password='123').status_code, 400)
        for _ in range(5):
            self.assertEqual(self.confirm(otp='999999').status_code, 400)
        self.assertEqual(self.confirm().status_code, 400)
        PasswordResetOTP.objects.filter(user=self.user).update(attempts=0, created_at=timezone.now() - timedelta(minutes=11))
        self.assertEqual(self.confirm().status_code, 400)
        self.user.refresh_from_db()
        self.assertTrue(self.user.check_password('Old-password-123'))

    @patch('user.views.send_mail', side_effect=OSError('SMTP unavailable'))
    def test_mail_failure_rolls_back_code(self, _send_mail):
        self.assertEqual(self.request_code().status_code, 503)
        self.assertFalse(PasswordResetOTP.objects.exists())
