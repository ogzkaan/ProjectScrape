from catboost import CatBoostRegressor
from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from numpy import sqrt 
from sklearn.preprocessing import LabelEncoder
from .models import Productjson

def pricePred(pName):
    ####
    #data= pd.read_csv("C:\Project_Scrape\pj_export - productjson (5).csv")
    #data['aggregateRating.ratingValue'] = pd.to_numeric(data['aggregateRating.ratingValue'], errors='coerce')
    #data.rename(columns={"aggregaterating_type": "aggregaterating_@type","offers_type":"offers_@type","field_context":"@context","field_type":"@type","id":"@id","brand_type":"brand_@type"}, inplace = True)
    #x=data.drop(["offers_price"],axis=1)
    #y=data['offers_price']
    """np.seterr(divide='ignore', invalid='ignore')
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    ####
    train=x_train.join(y_train)

    test=x_test.join(y_test)
    sub=test['offers_price']
    sub.columns=["offers_price"]
    test=test.drop(["offers_price"],axis=1)"""


    #train.shape, test.shape, sub.shape
    ####
    np.seterr(divide='ignore', invalid='ignore')
    qSet=Productjson.objects.filter(name=pName).values()
    data=pd.DataFrame(list(qSet))
    
    data['aggregaterating_ratingvalue'] = pd.to_numeric(data['aggregaterating_ratingvalue'], errors='coerce')
    
    data.rename(columns={"aggregaterating_reviewcount":"aggregateRating.reviewCount","aggregaterating_ratingcount":"aggregateRating.ratingCount","aggregaterating_ratingvalue":"aggregateRating.ratingValue","aggregaterating_type": "aggregaterating_@type","offers_type":"offers_@type","field_context":"@context","field_type":"@type","id":"@id","brand_type":"brand_@type"}, inplace = True)
    sub=data['offers_price']
    
    #data=data.drop(["offers_price"],axis=1)
    
    #df = pd.concat([test,train],ignore_index=True)
    df=data
    df['dt_created'] = pd.to_datetime(df['dt_created'])
    df['Day'] = df['dt_created'].dt.day
    df['Month'] = df['dt_created'].dt.month
    df['Year'] = df['dt_created'].dt.year
    #df['Dayofweek'] = pd.to_datetime(df['Date']).dt.dayofweek
    #df['DayOfyear'] = pd.to_datetime(df['Date']).dt.dayofyear
    #df['WeekOfyear'] = pd.to_datetime(df['Date']).dt.weekofyear
    df['Is_month_start'] =  df['dt_created'].dt.is_month_start
    df['Is_month_end'] = df['dt_created'].dt.is_month_end
    df['Is_quarter_start'] = df['dt_created'].dt.is_quarter_start
    df['Is_quarter_end'] = df['dt_created'].dt.is_quarter_end
    df['Is_year_start'] = df['dt_created'].dt.is_year_start
    df['Is_year_end'] = df['dt_created'].dt.is_year_end
    ####
    calc = df.groupby(['brandname'], axis=0).agg({'brandname':[('op1', 'count')]}).reset_index()
    calc.columns = ['brandname','brandname Count']
    df = df.merge(calc, on=['brandname'], how='left')

    calc = df.groupby(['kategori'], axis=0).agg({'kategori':[('op1', 'count')]}).reset_index()
    calc.columns = ['kategori','kategori Count']
    df = df.merge(calc, on=['kategori'], how='left')
    ####
    agg_func = {
        'aggregateRating.ratingValue': ['mean','min','max','sum']
    }

    agg_func = df.groupby('brandname').agg(agg_func)
    agg_func.columns = [ 'brandname' + ('_'.join(col).strip()) for col in agg_func.columns.values]
    agg_func.reset_index(inplace=True)
    df = df.merge(agg_func, on=['brandname'], how='left')

    agg_func = {
        'aggregateRating.ratingValue': ['mean','min','max','sum']
    }
    agg_func = df.groupby('kategori').agg(agg_func)
    agg_func.columns = [ 'kategori' + ('_'.join(col).strip()) for col in agg_func.columns.values]
    agg_func.reset_index(inplace=True)
    df = df.merge(agg_func, on=['kategori'], how='left')
    ####
    for c in ['brandname', 'kategori',]:
        df[c] = df[c].astype('category')
    ####
    agg_func = {
        'aggregateRating.ratingValue': ['mean','min','max','sum']
    }
    agg_func = df.groupby(['brandname', 'kategori']).agg(agg_func,use_numeric=True)
    agg_func.columns = [ 'brandname_kategori' + ('_'.join(col).strip()) for col in agg_func.columns.values]
    agg_func.reset_index(inplace=True)
    df = df.merge(agg_func, on=['brandname', 'kategori'], how='left')   
    ####
    df.drop(['aggregaterating_@type','offers_availability','offers_itemcondition','offers_pricecurrency','offers_url','offers_@type','url','dt_created','name','@context','@type','@id','image','description','sku','gtin13','brand_@type'], axis=1, inplace=True)
    ####
    train_df = df[df['offers_price'].isnull()!=True]
    test_df = df[df['offers_price'].isnull()==True]
    df.drop(['offers_price'], axis=1, inplace=True)
    ####
    train_df['offers_price'] = np.log1p(train_df['offers_price'].astype(float))
    ####
    X = train_df.drop(labels=['offers_price'], axis=1)
    y = train_df['offers_price'].values
    ####
    Xtest = df
    
    ####
    y_pred_totcat = []
    errcat = []
    cat = CatBoostClassifier()
    categorical_features_indices = np.where(X.dtypes == 'category')[0]
    cat_features=categorical_features_indices
    label_encoder = LabelEncoder()
    cat = CatBoostClassifier()
    cat = CatBoostRegressor(loss_function='RMSE',
                            eval_metric='RMSE',
                            depth=7,
                            random_seed=42,
                            iterations=1000,
                            learning_rate=0.1,
                            leaf_estimation_iterations=1,
                            l2_leaf_reg=1,
                            bootstrap_type='Bayesian',
                            bagging_temperature=1,
                            random_strength=1,
                            od_type='Iter',
                            od_wait=200)
                

    cat.load_model('C:\Project_Scrape\scrape\cat', format='cbm')
    #y_pred_cat = cat.predict(Xtest)

    Xtest.drop(['field_id'], axis=1, inplace=True)
    #Xtest.drop(['kategori'], axis=1, inplace=True)
    p = cat.predict(Xtest)
    #errcat.append(sqrt(mean_squared_log_error(np.exp(y_test), np.exp(y_pred_cat))))

    y_pred_totcat.append(p)
    np.mean(errcat,0)
    final = np.exp(np.mean(y_pred_totcat,0))
    sub['offers_price'] = final
    return sub