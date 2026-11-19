from django.urls import path

from .views import RegisterView, MeView, PasswordResetRequestView, PasswordResetConfirmView
from rest_framework_simplejwt.views import TokenBlacklistView, TokenObtainPairView, TokenRefreshView
urlpatterns = [
    path("register/", RegisterView.as_view(), name="dang-ky"),
    path("password-reset/request/", PasswordResetRequestView.as_view(), name="password-reset-request"),
    path("password-reset/confirm/", PasswordResetConfirmView.as_view(), name="password-reset-confirm"),
    path("me/",MeView.as_view(), name='Xem thong tin tai khoan' ),
    path("login/",TokenObtainPairView.as_view()),
    path("login/refresh/",  TokenRefreshView.as_view()),
    path("logout/", TokenBlacklistView.as_view() )
]
