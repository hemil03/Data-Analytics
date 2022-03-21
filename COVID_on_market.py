import pandas as pd
import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries

#defining alpha-vantage API key
api = '888888888'

#collecting data from API/url
deaths = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv")
confirm = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv")
american_markert = pd.read_csv('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=IBM&outputsize=full&apikey=' + api + '&datatype=csv')
Canadian_markert = pd.read_csv('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=IBM&outputsize=full&apikey=' + api + '&datatype=csv')
Travel_sector = pd.read_csv('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=LUV&outputsize=full&apikey=' + api + '&datatype=csv')
Real_Estate = pd.read_csv( 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=KIM&outputsize=full&apikey=' + api + '&datatype=csv')
silver = pd.read_csv('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=SIL&outputsize=full&apikey=' + api + '&datatype=csv')

#total confirmed cases
total_c = confirm.sum(axis = 0, skipna = True)
totalc = total_c.to_list()
#total confirmed deaths
total_d = deaths.sum(axis = 0, skipna = True)
totald = total_d.to_list()
#converting date, month, year 
timestamp = deaths.columns.to_list()
#removing unnecessary data
del timestamp[0:4]
del totalc[0:3]
del totald[0:3]

#data categorization/classification on covid dataframe
covid_cd = pd.DataFrame(list(zip(timestamp,totalc,totald)), columns = ['Timestamp', 'Total_Confirmed_Cases', 'Total_Deaths'])
covid_cd['Total_Confirmed_Cases'] = covid_cd.Total_Confirmed_Cases.astype('int')
covid_cd['Total_Deaths'] = covid_cd.Total_Deaths.astype('int')
covid_cd['Timestamp'] = pd.to_datetime(covid_cd['Timestamp'])

#data categorization/classification on american market data
american_markert['timestamp'] = pd.to_datetime(american_markert['timestamp'])
a_mark = american_markert.loc[(american_markert['timestamp'] >= '2020-01-22') & (american_markert['timestamp'] <= '2021-03-06')]
a_market = a_mark.sort_values(by='timestamp',ascending = True)

ibm_ts = a_mark['timestamp'].to_list()
ibm_high = a_mark['high'].to_list()
ibm_low = a_mark['low'].to_list()

IBM_data = pd.DataFrame(list(zip(ibm_ts,ibm_high,ibm_low)), columns = ['Timestamp', 'IBM_High', 'IBM_Low'])
final_1 = covid_cd.merge(IBM_data,how='outer',left_on=['Timestamp'],right_on=["Timestamp"])
final_1['American Market'] = 'IBM'

#data categorization/classification on canadian market data
Canadian_markert['timestamp'] = pd.to_datetime(Canadian_markert['timestamp'])
c_mark = Canadian_markert.loc[(Canadian_markert['timestamp'] >= '2020-01-22') & (Canadian_markert['timestamp'] <= '2021-03-06')]
c_market = c_mark.sort_values(by='timestamp',ascending = True)

shop_ts = c_mark['timestamp'].to_list()
shop_high = c_mark['high'].to_list()
shop_low = c_mark['low'].to_list()

SHOP_data = pd.DataFrame(list(zip(shop_ts,shop_high,shop_low)), columns = ['Timestamp', 'SHOP_High', 'SHOP_Low'])
final_2 = final_1.merge(SHOP_data,how='outer',left_on=['Timestamp'],right_on=["Timestamp"])
final_2['Canadian Market'] = 'SHOP'

#data categorization/classification on travel sector data
Travel_sector['timestamp'] = pd.to_datetime(Travel_sector['timestamp'])
Travel = Travel_sector.loc[(Travel_sector['timestamp'] >= '2020-01-22') & (Travel_sector['timestamp'] <= '2021-03-06')]
Travel = Travel.sort_values(by='timestamp',ascending = True)

LUV_ts = Travel['timestamp'].to_list()
LUV_high = Travel['high'].to_list()
LUV_low = Travel['low'].to_list()

Travel_data = pd.DataFrame(list(zip(LUV_ts,LUV_high,LUV_low)), columns = ['Timestamp', 'LUV_High', 'LUV_Low'])

final_3 = final_2.merge(Travel_data,how='outer',left_on=['Timestamp'],right_on=["Timestamp"])
final_3['Travel'] = 'LUV'

#data categorization/classification on real estate data
Real_Estate['timestamp'] = pd.to_datetime(Real_Estate['timestamp'])
ree = Real_Estate.loc[(Real_Estate['timestamp'] >= '2020-01-22') & (Real_Estate['timestamp'] <= '2021-03-06')]
rea = ree.sort_values(by='timestamp',ascending = True)

r_ts = rea['timestamp'].to_list()
r_high = rea['high'].to_list()
r_low = rea['low'].to_list()
KIM_data = pd.DataFrame(list(zip(r_ts,r_high,r_low)), columns = ['Timestamp','KIM_High', 'KIM_Low'])

final_4 = final_3.merge(KIM_data,how='outer',left_on=['Timestamp'],right_on=["Timestamp"])
final_4['Real Estate'] = 'KIM'

#data categorization/classification on silver metal data
silver['timestamp'] = pd.to_datetime(silver['timestamp'])
sil = silver.loc[(silver['timestamp'] >= '2020-01-22') & (silver['timestamp'] <= '2021-03-06')]
sil1 = sil.sort_values(by='timestamp',ascending = True)

s_ts = sil1['timestamp'].to_list()
s_high = sil1['high'].to_list()
s_low = sil1['low'].to_list()
met_data = pd.DataFrame(list(zip(s_ts,s_high,s_low)), columns = ['Timestamp', 'SIL_High', 'SIL_Low'])

#merging all final datasets into one
final_5 = final_4.merge(met_data,how='outer',left_on=['Timestamp'],right_on=["Timestamp"])
final_5['Metal'] = 'Silver'
#defining columns for final dataset
final_cols = ["Timestamp","Total_Confirmed_Cases","Total_Deaths","American Market","IBM_High","IBM_Low","Canadian Market","SHOP_High","SHOP_Low","Travel","LUV_High","LUV_Low","Real Estate","KIM_High","KIM_Low","Metal","SIL_High","SIL_Low"]
final_data = final_5.reindex(columns = final_cols)
#exporting final dataset
final_data.to_csv('COVID_effects_on_market.csv')
#find average from different sectors
f_ts = final_data['Timestamp'] 
f_tcs = (final_data['Total_Confirmed_Cases'] - final_data['Total_Confirmed_Cases'].mean()) / final_data['Total_Confirmed_Cases'].std()
f_tds = (final_data['Total_Deaths'] - final_data['Total_Deaths'].mean()) / final_data['Total_Deaths'].std()
fa_high = (final_data['IBM_High'] - final_data['IBM_High'].mean()) / final_data['IBM_High'].std()
fa_low = (final_data['IBM_Low'] - final_data['IBM_Low'].mean()) / final_data['IBM_Low'].std()
fc_high = (final_data['SHOP_High'] - final_data['SHOP_High'].mean()) / final_data['SHOP_High'].std()
fc_low = (final_data['SHOP_Low'] - final_data['SHOP_Low'].mean()) / final_data['SHOP_Low'].std()
ft_high = (final_data['LUV_High'] - final_data['LUV_High'].mean()) / final_data['LUV_High'].std()
ft_low = (final_data['LUV_Low'] - final_data['LUV_Low'].mean()) / final_data['LUV_Low'].std()
fr_high = (final_data['KIM_High'] - final_data['KIM_High'].mean()) / final_data['KIM_High'].std()
fr_low = (final_data['KIM_Low'] - final_data['KIM_Low'].mean()) / final_data['KIM_Low'].std()
fm_high = (final_data['SIL_High'] - final_data['SIL_High'].mean()) / final_data['SIL_High'].std()
fm_low = (final_data['SIL_Low'] - final_data['SIL_Low'].mean()) / final_data['SIL_Low'].std()
#plotting different graphs representing effect of COVID on different sectors
plt.plot(f_ts, f_tcs, label = 'Confirmed Cases')
plt.plot(f_ts, f_tds, label = 'Deaths')
plt.title('Total Confirmed Cases vs. Total Deaths')
plt.xlabel('Timeline')
plt.ylabel('Frequency')
plt.legend()
plt.show()

plt.plot(f_ts, fa_high, label = 'High')
plt.plot(f_ts, fa_low, label = 'Low')
plt.title('High vs. Low')
plt.xlabel('Timeline')
plt.ylabel('Frequency')
plt.legend()
plt.show()

plt.plot(f_ts, f_tcs, label = 'Cases')
plt.plot(f_ts, f_tds, label = 'Deaths')
plt.plot(f_ts, fa_high, label = 'High')
plt.plot(f_ts, fa_low, label = 'Low')
plt.title('Overall')
plt.xlabel('Timeline')
plt.ylabel('Frequency')
plt.legend()
plt.show()

plt.plot(f_ts, fc_high, label = 'High')
plt.plot(f_ts, fc_low, label = 'Low')
plt.title('High vs. Low')
plt.xlabel('Timeline')
plt.ylabel('Frequency')
plt.legend()
plt.show()

plt.plot(f_ts, f_tcs, label = 'Cases')
plt.plot(f_ts, f_tds, label = 'Deaths')
plt.plot(f_ts, fc_high, label = 'High')
plt.plot(f_ts, fc_low, label = 'Low')
plt.title('Overall')
plt.xlabel('Timeline')
plt.ylabel('Frequency')
plt.legend()
plt.show()

plt.plot(f_ts, ft_high, label = 'High')
plt.plot(f_ts, ft_low, label = 'Low')
plt.title('High vs. Low')
plt.xlabel('Timeline')
plt.ylabel('Frequency')
plt.legend()
plt.show()

plt.plot(f_ts, f_tcs, label = 'Cases')
plt.plot(f_ts, f_tds, label = 'Deaths')
plt.plot(f_ts, ft_high, label = 'High')
plt.plot(f_ts, ft_low, label = 'Low')
plt.title('Overall')
plt.xlabel('Timeline')
plt.ylabel('Frequency')
plt.legend()
plt.show()

plt.plot(f_ts, fr_high, label = 'High')
plt.plot(f_ts, fr_low, label = 'Low')
plt.title('Real Estate High vs. Low')
plt.xlabel('Timeline')
plt.ylabel('Frequency')
plt.legend()
plt.show()

plt.plot(f_ts, f_tcs, label = 'Confirmed Cases')
plt.plot(f_ts, f_tds, label = 'Deaths')
plt.plot(f_ts, fr_high, label = 'High')
plt.plot(f_ts, fr_low, label = 'Low')
plt.title('Overall Real Estate Data with COVID')
plt.xlabel('Timeline')
plt.ylabel('Frequency')
plt.legend()
plt.show()

plt.plot(f_ts, fm_high, label = 'High')
plt.plot(f_ts, fm_low, label = 'Low')
plt.title('High vs. Low')
plt.xlabel('Timeline')
plt.ylabel('Frequency')
plt.legend()
plt.show()

plt.plot(f_ts, f_tcs, label = 'Cases')
plt.plot(f_ts, f_tds, label = 'Deaths')
plt.plot(f_ts, fm_high, label = 'High')
plt.plot(f_ts, fm_low, label = 'Low')
plt.title('Overall')
plt.xlabel('Timeline')
plt.ylabel('Frequency')
plt.legend()
plt.show()