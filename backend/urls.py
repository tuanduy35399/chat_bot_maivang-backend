
from django.contrib import admin
from django.urls import path, include
from drf_spectacular.views import SpectacularAPIView, SpectacularSwaggerView

urlpatterns = [
    path('admin/', admin.site.urls,),
    path('api/v1/user/', include("user.urls"),),
    path('api/v1/diseases/', include("diseases.urls"),),
    path('api/v1/schema/', SpectacularAPIView.as_view(),name='schema'), #buoc phai dat name='schema'
    path('api/v1/docs/', SpectacularSwaggerView.as_view(),),
    path('api/v1/history/', include("history.urls")),
    # path("api-auth/", include("rest_framework.urls"))
]
