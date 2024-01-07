# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Make sure each ForeignKey and OneToOneField has `on_delete` set to the desired behavior
#   * Remove `managed = False` lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.
from django.db import models




class Users(models.Model):
    idusers = models.AutoField(db_column='idUsers', primary_key=True)  # Field name made lowercase.
    username = models.CharField(db_column='Username', unique=True, max_length=45, blank=True, null=True)  # Field name made lowercase.
    password = models.CharField(db_column='Password', max_length=45, blank=True, null=True)  # Field name made lowercase.

    class Meta:
        managed = False
        db_table = 'users'



class Productjson(models.Model):
    field_context = models.CharField(db_column='@context', max_length=400, blank=True, null=True)  # Field renamed to remove unsuitable characters. Field renamed because it started with '_'.
    field_type = models.CharField(db_column='@type', max_length=400, blank=True, null=True)  # Field renamed to remove unsuitable characters. Field renamed because it started with '_'.
    field_id = models.CharField(db_column='@id', max_length=400, blank=True, null=True)  # Field renamed to remove unsuitable characters. Field renamed because it started with '_'.
    name = models.CharField(max_length=400, blank=True, null=True)
    image = models.CharField(max_length=400, blank=True, null=True)
    description = models.CharField(max_length=400, blank=True, null=True)
    sku = models.CharField(max_length=400, blank=True, null=True)
    gtin13 = models.CharField(max_length=400, blank=True, null=True)
    brand_type = models.CharField(db_column='brand.@type', max_length=400, blank=True, null=True)  # Field renamed to remove unsuitable characters.
    brandname = models.CharField(max_length=400, blank=True, null=True)
    url = models.CharField(max_length=400, blank=True, null=True)
    offers_type = models.CharField(db_column='offers.@type', max_length=400, blank=True, null=True)  # Field renamed to remove unsuitable characters.
    offers_url = models.CharField(db_column='offers.url', max_length=400, blank=True, null=True)  # Field renamed to remove unsuitable characters.
    offers_pricecurrency = models.CharField(db_column='offers.priceCurrency', max_length=400, blank=True, null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters.
    offers_price = models.CharField(db_column='offers.price', max_length=400, blank=True, null=True)  # Field renamed to remove unsuitable characters.
    offers_itemcondition = models.CharField(db_column='offers.itemCondition', max_length=400, blank=True, null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters.
    offers_availability = models.CharField(db_column='offers.availability', max_length=400, blank=True, null=True)  # Field renamed to remove unsuitable characters.
    aggregaterating_type = models.CharField(db_column='aggregateRating.@type', max_length=400, blank=True, null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters.
    aggregaterating_ratingvalue = models.CharField(db_column='aggregateRating.ratingValue', max_length=20, blank=True, null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters.
    aggregaterating_ratingcount = models.CharField(db_column='aggregateRating.ratingCount', max_length=20, blank=True, null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters.
    aggregaterating_reviewcount = models.CharField(db_column='aggregateRating.reviewCount', max_length=20, blank=True, null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters.
    dt_created = models.DateTimeField(blank=True, null=True)
    kategori = models.CharField(max_length=45, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'productjson'

class Kategoritablosu(models.Model):
    checkbox_id = models.AutoField(db_column='checkbox_id', primary_key=True)  # Field name made lowercase.
    name = models.CharField(max_length=45, blank=True, null=True)
    is_checked = models.IntegerField(blank=True, null=True)
    pazaryeri = models.CharField(max_length=45, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'kategoritablosu'

class Inventory(models.Model):
    idinventory = models.AutoField(db_column='idInventory', primary_key=True)  # Field name made lowercase.
    sku = models.CharField(db_column='SKU', max_length=45, blank=True, null=True)  # Field name made lowercase.
    product_name = models.CharField(db_column='ProductName', max_length=45, blank=True, null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters.
    product_brand = models.CharField(db_column='ProductBrand', max_length=45, blank=True, null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters.
    purchase = models.CharField(db_column='Purchase', max_length=45, blank=True, null=True)  # Field name made lowercase.
    kdv = models.CharField(db_column='KDV', max_length=45, blank=True, null=True)  # Field name made lowercase.

    class Meta:
        managed = False
        db_table = 'inventory'