from django.urls import path
from . import views

urlpatterns=[
    path('',views.index,name='dashboard'),
    path('detay/',views.detayPage,name='detay'),
    path('detay/brands/',views.brands,name='brands'),
    path('detay/kategoriler/',views.kategoriler,name='kategoriler'),
    path('detay/fiyatHesaplama/',views.fiyatHesaplama,name='fiyatHesaplama'),
    path('detay/urunEkleme/',views.urunEkleme,name='urunEkleme'),
    
    

]
"""path('detay/brandDetay/',views.brandDetay,name='brandDetay'),
    path('detay/kategoriDetay/',views.kategoriDetay,name='kategoriDetay'),"""