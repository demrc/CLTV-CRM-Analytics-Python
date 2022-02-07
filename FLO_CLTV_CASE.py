!pip install lifetimes
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
from sklearn.preprocessing import MinMaxScaler

df_=pd.read_csv(r"....csv")

df = df_.copy()

#Part 1

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = round(quartile3 + 1.5 * interquantile_range)
    low_limit = round(quartile1 - 1.5 * interquantile_range)
    
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

#--------------------------
df.isnull().sum()

replace_with_thresholds(df, "order_num_total_ever_online")

replace_with_thresholds(df, "order_num_total_ever_offline")

replace_with_thresholds(df, "customer_value_total_ever_offline")

replace_with_thresholds(df, "customer_value_total_ever_online")

df["omnichannel_order"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["omnichannel_order"]

df["omnichannel_value"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
df["omnichannel_value"]

df["first_order_date"]= pd.to_datetime(df["first_order_date"], format="%Y-%m-%d" )
df["last_order_date"]= pd.to_datetime(df["last_order_date"], format="%Y-%m-%d" )
df["last_order_date_offline"]= pd.to_datetime(df["last_order_date_offline"], format="%Y-%m-%d" )
df["last_order_date_online"]= pd.to_datetime(df["last_order_date_online"], format="%Y-%m-%d" )

#Part 2

df["new_day"]=(df["last_order_date"]-df["first_order_date"])
df["new_day"].isnull().sum()

df["last_order_date"].max()

today_date = dt.datetime(2021, 6, 1)

cltv_df = df.groupby('master_id').agg(
    {'new_day': lambda new_day: (new_day).astype('timedelta64[D]'),
     'last_order_date':lambda last_date:(today_date-last_date).astype('timedelta64[D]'),
     'omnichannel_order': lambda order: order,
     'omnichannel_value': lambda value: value.sum()})

cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']

cltv_df=cltv_df[cltv_df["T"]>1]
cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]
cltv_df = cltv_df[(cltv_df['frequency'] > 1)]
cltv_df = cltv_df[(cltv_df['recency'] > 0)]
cltv_df = cltv_df[(cltv_df['T'] > cltv_df["recency"])]
cltv_df["recency"] = cltv_df["recency"] / 7
cltv_df.describe().T
cltv_df["T"] = cltv_df["T"] / 7


#Part 3

bgf = BetaGeoFitter(penalizer_coef=0.01)

bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T'])

cltv_df["exp_sales_3_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(4*3,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T'])

cltv_df["exp_sales_3_month"].sort_values(ascending=False).head(10)

plot_period_transactions(bgf)
plt.show()

cltv_df["exp_sales_6_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(4*6,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T'])
cltv_df["exp_sales_6_month"].sort_values(ascending=False).head(10)

plot_period_transactions(bgf)
plt.show()


#Her iki model arasında ciddi bir fark yoktur. Farklı olduğu alanlar ise çok ufak farklı yerlerdir. Örneğin 5. zamanda 3 ay
#için model gerçek veriyi geçerken 6 ay için gerçek veri modeli geçmiştir.

ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv_df['frequency'], cltv_df['monetary'])

cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary'])

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=6,  # 6 aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)

cltv = cltv.reset_index() 

scaled_cltv = cltv_df.merge(cltv, on="master_id", how="left")


cltv.sort_values(by="clv",ascending=False).head(20)

#Part 4

scaled_cltv["segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])
scaled_cltv["segment"]

scaled_cltv.groupby("segment").agg({
    "count","mean","sum"
})
