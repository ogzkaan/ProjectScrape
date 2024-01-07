from django import forms
from .models import Kategoritablosu

class KategoriForm(forms.ModelForm):
    class Meta:
        model = Kategoritablosu
        fields = ['is_checked']