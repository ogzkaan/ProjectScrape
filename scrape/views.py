from django.shortcuts import render, HttpResponse,redirect
from django.http import JsonResponse
from django.contrib import messages
from django.contrib.auth.models import User
from .models import Productjson
from .models import Kategoritablosu
from .forms import KategoriForm
from .scrape import scrape_
from .testt import pricePred
from .db import *
from django.http import JsonResponse
from django.core.serializers import serialize
import json
# Create your views here.
def  index(request):
    productjson=Productjson.objects.values("name","url","brandname","kategori","offers_price").distinct()
    context={'productjson':productjson}
    if 'scrape' in request.POST:
        if request.method == 'POST':
            data=request.POST["sayfasayisi"],request.POST["kategori"],request.POST["pazaryeri"]
            if data is not None:
                scrape_(data[0],data[1],data[2])
                return redirect('dashboard')
            else:
                return render (request,'accounts/login.html', {'error':'err'})
    elif 'detayPage' in request.POST:
         if request.method == 'POST':
            data=request.POST["detayPage"]
            print(data)
            pj_OneRow=Productjson.objects.filter(name=data).values()
            for item in pj_OneRow:
                item['image_urls'] = item['image'].split(',') 
            context1={'productjson':pj_OneRow}
            return render(request,"detay.html",context1)
    elif 'brandDetay'in request.POST:
        if request.method == 'POST':
            data=request.POST["brandDetay"]
            context=Productjson.objects.filter(brandname=data).values("name","url","kategori","offers_price").distinct()
            context={'productjson':context}
            return render(request,"brandDetay.html",context) 
    elif 'kategoriDetay'in request.POST:
        if request.method == 'POST':
            data=request.POST["kategoriDetay"]
            context=Productjson.objects.filter(kategori=data).values("name","url","brandname","offers_price").distinct()
            context={'productjson':context}
            return render(request,"kategoriDetay.html",context)
    else:
            return render(request,"index.html",context)

def detayPage(request):
    return render(request,"detay.html")
def fiyatHesaplama(request):
    return render(request,"fiyatHesaplama.html")
def urunEkleme(request):
    if 'referansBul'in request.POST:
        if request.method == 'POST':
            data=request.POST["referansAdı"]
            predPrice=pricePred(data)
            pj_OneRow=Productjson.objects.filter(name=data).values()
            for item in pj_OneRow:
                item['image_urls'] = item['image'].split(',') 
            context1={'productjson':pj_OneRow}
            context1.update({"pred":predPrice['offers_price']})
            print(context1)
            return render(request,"urunEkleme.html",context1)
    elif 'ürünEkle'in request.POST:
        if request.method == 'POST':
            data=request.POST["SKU"],request.POST["ürünAdi"],request.POST["marka"],request.POST["alisFiyati"],request.POST["kdv"]
            print(data)
            
            filtered_list = list(filter(lambda x: x, data))
            
            if not filtered_list :
                print("Veri giriniz")
                
            else:
                insertInventory(data[0],data[1],data[2],data[3],data[4])  
    else:
            return render(request,"urunEkleme.html")
                
    return render(request,"urunEkleme.html")
def brands(request):
    if 'detayPage' in request.POST:
         if request.method == 'POST':
            data=request.POST["detayPage"]
            print(data)
            pj_OneRow=Productjson.objects.filter(name=data).values()
            for item in pj_OneRow:
                item['image_urls'] = item['image'].split(',') 
            context1={'productjson':pj_OneRow}
            return render(request,"detay.html",context1)
    elif 'filtre' in request.POST:
        if request.method == 'POST':
            print(request.POST)
            data=request.POST["filtre"]
            print(data)
            brandnames=Productjson.objects.values("brandname")
            context=Productjson.objects.filter(brandname=data).values("name","url","kategori","offers_price","brandname").distinct()

            #context=(brandnames,context1)
            context={'productjson':context}
            print(context)
            return render(request,"brands.html",context)
    else:
        #data=request.POST["brands"]
        context=Productjson.objects.values("name","url","kategori","brandname","offers_price")
        context={'productjson':context}
        return render(request,"brands.html",context)


def kategoriler(request):
    print(request.POST)
    if 'detayPage' in request.POST:
         if request.method == 'POST':
            data=request.POST["detayPage"]
            pj_OneRow=Productjson.objects.filter(name=data).values()
            for item in pj_OneRow:
                item['image_urls'] = item['image'].split(',') 
            context1={'productjson':pj_OneRow}
            return render(request,"detay.html",context1)
    if 'filtre' in request.POST:
        if request.method == 'POST':
            data=request.POST["filtre"]
            context=Productjson.objects.filter(kategori=data).values("name","url","kategori","offers_price","brandname").distinct()

            #context=(brandnames,context1)
            context={'productjson':context}
            print(context)
            return render(request,"kategoriler.html",context)
    else:
        #data=request.POST["brands"]
        context=Productjson.objects.values("name","url","kategori","brandname","offers_price")
        context={'productjson':context}
        return render(request,"kategoriler.html",context)