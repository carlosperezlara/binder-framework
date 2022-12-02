import pandas
import numpy
import math

from datetime import datetime, date

import scipy
import sklearn.linear_model
import sklearn.metrics
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

class magic:
    def __init__(self):
        def Date2Unique(df,postTrim=True):
            df['Unique'] = (df['Date']-pandas.Timestamp('19000101')).dt.days
            if postTrim:
                df.drop(columns=['Date'],inplace=True)
            return
        def Unique2GenDate(df):
            df['Date'] = df['Unique'].apply( lambda x: pandas.Timedelta(x, unit="D") + date.fromisoformat('1900-01-01') )
            return
        def Date2YM(df,postTrim=True):
            df['YM'] = df['Date'].dt.year*100 + df['Date'].dt.month
            if postTrim:
                df.drop(columns=['Date'],inplace=True)
            return

        #HISTORIC PRICE OF HEATING OIL PER WEEK
        self.df_dpg_ny = pandas.read_csv("price_ny.csv",dtype={"Date":"str"})
        self.df_dpg_ny["Date"] = pandas.to_datetime(self.df_dpg_ny["Date"])
        Date2YM(self.df_dpg_ny,False)

        #HISTORIC TEMPERATURE IN LONG ISLAND PER WEEK
        self.df_temp_ny = pandas.read_csv("temp_ny_2.csv")
        self.df_temp_ny["Date"] = pandas.to_datetime(self.df_temp_ny["datetime"])
        Date2Unique(self.df_temp_ny,True)
        
        #HISTORIC SUPPLY OF CRUDE OIL
        self.df_imp_ny = pandas.read_csv("imp_ny.csv",dtype={"Date":"str"})
        self.df_imp_nj = pandas.read_csv("imp_nj.csv",dtype={"Date":"str"})
        self.df_pro_ny = pandas.read_csv("pro_ny.csv",dtype={"Date":"str"})
        self.df_imp_ny["Date"] = pandas.to_datetime(self.df_imp_ny["Date"])
        self.df_imp_nj["Date"] = pandas.to_datetime(self.df_imp_nj["Date"])
        self.df_pro_ny["Date"] = pandas.to_datetime(self.df_pro_ny["Date"])
        #GENERATE UNIQUE
        Date2Unique(self.df_pro_ny,False)
        Date2Unique(self.df_imp_ny,True)
        Date2Unique(self.df_imp_nj,True)
        #MERGE ALL SOURCES VIA UNIQUE
        self.df_all = self.df_pro_ny[['Unique','Date','Barrels']]
        self.df_all = self.df_all.join( self.df_imp_ny.set_index('Unique'), on="Unique", how="outer", rsuffix="ImpNY" )
        self.df_all = self.df_all.join( self.df_imp_nj.set_index('Unique'), on="Unique", how="outer", rsuffix="ImpNJ" )
        #REGENERATE DATE
        Unique2GenDate( self.df_all )
        self.df_all["Date"] = pandas.to_datetime(self.df_all["Date"])
        self.df_all.sort_values(by='Unique', inplace=True)
        #DISPLACE BY A MONTH
        self.df_all["Date"] = self.df_all["Date"].shift(periods=-1)
        Date2YM(self.df_all,True)
        #DROP UNIQUE
        self.df_all.drop(columns=['Unique'],inplace=True)

        #MERGE PRICE AND SUPPLY VIA YEARMONTH
        self.df_all = self.df_dpg_ny.join( self.df_all.set_index('YM'), on='YM', how='left', rsuffix='feats' )
        Date2Unique(self.df_all,True)
        #print( self.df_all.head() )
        #print( self.df_temp_ny.tail() )
        self.df_all = self.df_temp_ny.join( self.df_all.set_index('Unique'), on='Unique', how='right', rsuffix='temp' )
        #self.df_all['temp'].fillna(value=0)

        #GENERATE UNIQUE BASED ON PRICE TABLE
        Unique2GenDate( self.df_all )
        self.df_all["Date"] = pandas.to_datetime(self.df_all["Date"])
        #print( self.df_all.head() )

        return
    
    def selectYear(self,year):
        #FILTER ON WINTER SEASON
        filter0 = self.df_all["Date"].dt.strftime("%Y").astype(int) == year   ## year
        filter1 = self.df_all["Date"].dt.strftime("%Y").astype(int) == year+1 ## year+1
        filter2 = self.df_all["Date"].dt.month < 4 ## April
        filter3 = self.df_all["Date"].dt.month > 9 ## September
        self.selected = self.df_all[(filter0&filter3)|(filter1&filter2)].copy()
        #SORT AND ADD PRICE ANCESTRY
        self.selected.sort_values(by='Date', inplace=True)
        self.selected['DPG1'] = self.selected['DPG'].shift(periods=1)
        #self.selected['DPG2'] = self.selected['DPG'].shift(periods=2)
        #self.selected['DPG3'] = self.selected['DPG'].shift(periods=3)
        #CHOP ORPHANS AND RESET INDEX
        #self.selected = self.selected.iloc[3:,:]
        self.selected = self.selected.iloc[1:,:]
        self.selected.reset_index(inplace=True)
        self.selected['time'] = self.selected['Unique'] - self.selected['Unique'].min()
        self.selected['time2'] = numpy.power(self.selected['time'],2)
        self.selected['time3'] = numpy.power(self.selected['time'],3)
        self.selected['time4'] = numpy.power(self.selected['time'],4)
        #print(self.selected.head())
        return
    
    def trainModel(self,partial_train=None):
        if self.selected is None:
            return

        x = self.selected[['time','time2','time3','time4','DPG1','Barrels','BarrelsImpNY',"BarrelsImpNJ",'temp','windspeed']]
        y = self.selected['DPG']
        #print( self.selected.head() )

        clf1 = sklearn.linear_model.LinearRegression()
        parameters2 = {'alpha':[0.05,0.1,0.3,0.6,1.0,3.0,10,20,30,100]}
        clf2 = GridSearchCV( sklearn.linear_model.Lasso(max_iter=10000,tol=0.01), parameters2 )
        parameters3 = {'max_depth':[1,2,3,4,5,6]}
        clf3 = GridSearchCV( DecisionTreeRegressor(), parameters3 )

        pred = []
        train = []
        if partial_train is not None:
            train = y.iloc[:partial_train]
            clf1.fit( x.iloc[:partial_train], y.iloc[:partial_train] )
            pred1 = clf1.predict( x.iloc[:partial_train+1] )
            clf2.fit( x.iloc[:partial_train], y.iloc[:partial_train] )
            pred2 = clf2.predict( x.iloc[:partial_train+1] )
            clf3.fit( x.iloc[:partial_train], y.iloc[:partial_train] )
            pred3 = clf3.predict( x.iloc[:partial_train+1] )
        else:
            train = y
            clf1.fit( x, y )
            pred1 = clf1.predict(x)
            clf2.fit( x, y )
            pred2 = clf2.predict(x)
            clf3.fit( x, y )
            pred3 = clf3.predict(x)
        self.selected['Trained'] = pandas.Series( train )

        self.selected['Predicted1'] = pandas.Series( pred1 )
        self.selected['Predicted2'] = pandas.Series( pred2 )
        self.selected['Predicted3'] = pandas.Series( pred3 )
        self.selected['PredictedAVG'] = (self.selected['Predicted1']+self.selected['Predicted2']+self.selected['Predicted3'])/3
        
        print("GridSeearchCV convergence")
        print("Lasso",clf2.best_params_)
        print("DecisionTree",clf3.best_params_)
        return 
    
    def getDB(self):
        return self.selected
    