from bs4 import BeautifulSoup
import requests
import numpy as np
from .db import *
import json


def scrape_(sayfa,kategori,pazaryeri):
    headers={
            "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/XXX.0.0.0 Safari/XXX.36"
        }
    for i in range(1,int(sayfa)):
        url=("https://www.trendyol.com/"+kategori+"?pi="+str(i))

        page = requests.get(url,headers=headers)

        htmlPagePListing=BeautifulSoup(page.content,"html.parser")
        data=[]
        
        for p in htmlPagePListing.findAll("div", class_="p-card-wrppr with-campaign-view add-to-bs-card"):
            anchor=p.find('a')
            href=anchor.get('href')
            #print(href)
            url1="https://www.trendyol.com/"+href

            page=requests.get(url1,headers=headers)
            htmlPagePDetail=BeautifulSoup(page.content,"html.parser")
            try:
                j=htmlPagePDetail.find("script", type="application/ld+json").get_text()
            except:
                print("err")
            productDetailJson=json.loads(j)
            print("tarama")
            try:
                data.extend([[productDetailJson["@context"]],[productDetailJson["@type"]],[productDetailJson["@id"]],[productDetailJson["name"]],[productDetailJson["image"]],[productDetailJson["description"]],[productDetailJson["sku"]],[productDetailJson["gtin13"]],[productDetailJson["brand"]["@type"]],[productDetailJson["brand"]["name"]],[productDetailJson["url"]],[productDetailJson["offers"]["@type"]],[productDetailJson["offers"]["url"]],[productDetailJson["offers"]["priceCurrency"]],[productDetailJson["offers"]["price"]],[productDetailJson["offers"]["itemCondition"]],[productDetailJson["offers"]["availability"]],[productDetailJson["aggregateRating"]["@type"]],[productDetailJson["aggregateRating"]["ratingValue"]],[productDetailJson["aggregateRating"]["ratingCount"]],[productDetailJson["aggregateRating"]["reviewCount"]]])
                data.extend([[kategori]])
                InsterProductJson(data)
            except:
                print("erroe")
            
            data.clear()
            






