import mysql.connector

db=mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="ogz123",
    database="UrunTablosu"
)


myCursor=db.cursor(buffered=True)

#myCursor.execute("CREATE TABLE UrunDetay(ProductName VARCHAR(200),ProductPrice float UNSIGNED,ProductID int PRIMARY KEY AUTO_INCREMENT)")
#myCursor.execute("DESCRIBE UrunDetay")

insertVendorQuery_="INSERT INTO  Vendors(VendorName,E_mail) VALUES(%s,%s)"
insertProductQuery_="INSERT IGNORE INTO  UrunDetay(ProductName,ProductPrice,ProductCategory) VALUES(%s,%s,%s)"
insertInventoryQuery_="INSERT IGNORE INTO  inventory(sku,ProductName,ProductBrand,Purchase,KDV) VALUES(%s,%s,%s,%s,%s)"
insertUserQuery_="INSERT IGNORE INTO  Users(Username,Password) VALUES(%s,%s)"
selectUserQuery_="Select * from Users where username = %s and password = %s"
selectProductQuery_="Select * from UrunDetay"
selectVendorQuery_="Select * from Vendors"
insertProductJsonQuery_="INSERT IGNORE INTO productjson(`@context`,`@type`,`@id`,name,image,description,sku,gtin13,`brand.@type`,brandname,url,`offers.@type`,`offers.url`,`offers.priceCurrency`,`offers.price`,`offers.itemCondition`,`offers.availability`,`aggregateRating.@type`,`aggregateRating.ratingValue`,`aggregateRating.ratingCount`,`aggregateRating.reviewCount`,`kategori`) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"

def InsterProductJson(data):
    data = [str(s).strip("[]'") for s in data]
    myCursor.execute(insertProductJsonQuery_,data)
    db.commit()
def insertInventory(sku,product_name,product_brand,purchase,kdv):
    myCursor.execute(insertInventoryQuery_,(sku,product_name,product_brand,purchase,kdv))
    db.commit()
def InsertProduct(ProductName,ProductPrice,ProductCategory):
    myCursor.execute(insertProductQuery_,(ProductName,ProductPrice,ProductCategory))
    db.commit()
def InsertVendor(VendorName,Email):
    myCursor.execute(insertVendorQuery_,(VendorName,Email))
    db.commit()
def InsertUser(Username,Password):
    myCursor.execute(insertUserQuery_,(Username,Password))
    db.commit()    
def SelectUser(Username,Password):
    myCursor.execute(selectUserQuery_,(Username,Password))
    return myCursor.fetchall()
def SelectProduct():
    myCursor.execute(selectProductQuery_)
    return myCursor.fetchall()
def SelectVendor():
    myCursor.execute(selectVendorQuery_)
    return myCursor.fetchall()