from django.shortcuts import render, HttpResponse,redirect
from django.contrib import auth
from django.shortcuts import render, HttpResponse,redirect
from django.contrib import messages
from django.contrib.auth.models import User
from django.http import HttpResponseRedirect 
from django.urls import reverse
# Create your views here.
def loginPage(request):
    if request.method == 'POST':
        user = auth.authenticate(username=request.POST['username'],password = request.POST['password'])
        print(user)
        if user is not None:
            auth.login(request,user)   
            return HttpResponseRedirect(reverse('dashboard'))
        else:
            return render (request,'login.html', {'error':'Username or password is incorrect!'})
    else:
        return render(request,'login.html')
def registerPage(request):
    if request.method=="POST":
        if request.POST['password'] == request.POST['passwordC']:
            try:
                User.objects.get(username=request.POST['Username'])
                return render(request,'register.html')
            except User.DoesNotExist:
                user=User.objects.create_user(username=request.POST['username'],email=request.POST['email'],password=request.POST['password'])
                auth.login(request,user)
                return HttpResponseRedirect(reverse('dashboard'))
        else:
            return render(request,'register.html',{'error':'Password does not match!'})
    else:
        return render(request,'register.html')