import pandas as pd
import numpy as np

from pandas_datareader import data as pdr
from tqdm import tqdm_notebook
import talib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import urllib.request
from google_drive_downloader import GoogleDriveDownloader as gdd


import streamlit as st
import datetime


# ใส่ title ของ sidebar
st.sidebar.write('## KBANK.BKK stock predictor')
st.sidebar.write('### Enter first and last date for prediction for KBANK.BKK stock')
first_date = st.sidebar.date_input('First date', datetime.date(2010,6,27))
last_date = st.sidebar.date_input('Last date', datetime.date(2021,6,27))
st.sidebar.write('If you want a predicted price of today\'s KBANK stock enter the current date.')


#main
st.write("""
# INPUT
"""
    )

st.write(first_date)
st.write("TO")
st.write(last_date)

st.write("""
# RESULT
"""
    )

stock_list = ['KBANK','SCB','BBL','KTB']
stock_data = []
stock_name = []
for quote in tqdm_notebook(stock_list):
    try:
        stock_data.append(pdr.get_data_yahoo(f'{quote}.BK', start=first_date, end=last_date))
        stock_name.append(quote)
    except:
        print("Error:", sys.exc_info()[0])
        print("Description:", sys.exc_info()[1])

stock_data1=stock_data
st.write("using :",device)



#preprocessing data

sma = talib.SMA(stock_data[0].Close, timeperiod=20)  # default period is 30, changed to 20
sma = sma.to_frame()
sma.columns = ['SMA']

t3 = talib.T3(stock_data[0].Close, timeperiod=5)
t3 = t3.to_frame()
t3.columns = ['t3']

stock_data1 = pd.concat([stock_data1[0],sma,t3],axis=1) #concat them






# function สำหรับ preprocess ข้อมูล time series หลายตัวแปร
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True,feat_name=None):
    
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [f'{feat_name[j]}(t-{i})' for j in range(n_vars)]
    
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [f'{feat_name[j]}(t)' for j in range(n_vars)]
        else:
            names += [f'{feat_name[j]}(t+{i})' for j in range(n_vars)]
    
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
    

# เลือกข้อมูลหุ้นจาก list ของ DataFrame และ drop column 'Close' เนื่องจากเราจะใช้ column 'Adj. Close' เท่านั้น
dataset1 = stock_data1
dataset1 = dataset1.drop('Close',axis=1)
dataset1['pct_change'] = dataset1['Adj Close'].pct_change().dropna()
values1 = dataset1.values
values1 = values1.astype('float32')

# ทำ scaling ข้อมูลด้วยวิธี min max scaling เปลี่ยน scale ข้อมูลแต่ละ column ให้อยู่ระหว่าง [0,1] และเก็บค่า min max แต่ละ column ไว้สำหรับทำ rescale ข้อมูลภายหลัง
#I think this part can be replaced with sigmoid
min_dict = dict()
max_dict = dict()
for col in dataset1.columns:
  min_dict[col] = dataset1[col].min()
  max_dict[col] = dataset1[col].max()
  dataset1[col] = (dataset1[col] - dataset1[col].min())/(dataset1[col].max()-dataset1[col].min())
  
# ใช้ function สำหรับ preprocess ข้อมูลที่เขียนไว้ และ drop column ที่ไม่ได้ใช้
reframed1 = series_to_supervised(dataset1.values, 30, 1,feat_name=dataset1.columns)

reframed1.drop(['High(t)','Low(t)','Open(t)','Volume(t)','Adj Close(t)','SMA(t)','t3(t)'],
              axis=1,inplace=True)#แก้ตรงนี้






# turn into test sets
n_train_percent = 0.0
split = int(reframed1.shape[0]*n_train_percent)
df_seq_test1 = reframed1
date_all = dataset1.index[reframed1.index]




df_dt = pd.DataFrame({'date_time':date_all})

df_dt_feat = pd.concat([df_dt.date_time.dt.day, df_dt.date_time.dt.dayofweek, df_dt.date_time.dt.dayofyear,\
                           df_dt.date_time.dt.daysinmonth, df_dt.date_time.dt.is_month_end,df_dt.date_time.dt.is_month_start,\
                           df_dt.date_time.dt.is_quarter_end, df_dt.date_time.dt.is_quarter_start, df_dt.date_time.dt.is_year_end,\
                           df_dt.date_time.dt.is_year_start],axis=1)
df_dt_feat.columns = ['day','dayofweek','dayofyear','daysinmonth','is_month_end','is_month_start','is_quarter_end','is_quarter_start',\
                        'is_year_end','is_year_start']

df_dt_feat['is_month_end'] = df_dt_feat['is_month_end'].astype(int)
df_dt_feat['is_month_start'] = df_dt_feat['is_month_start'].astype(int)
df_dt_feat['is_quarter_end'] = df_dt_feat['is_quarter_end'].astype(int)
df_dt_feat['is_quarter_start'] = df_dt_feat['is_quarter_start'].astype(int)
df_dt_feat['is_year_end'] = df_dt_feat['is_year_end'].astype(int)
df_dt_feat['is_year_start'] = df_dt_feat['is_year_start'].astype(int)

# encode categorical columns
for col in ['day','dayofweek','dayofyear','daysinmonth']:
  df_dt_feat[col] = df_dt_feat[col].astype('category').cat.as_ordered().cat.codes

df_dt_feat_train = df_dt_feat.iloc[:split]
df_dt_feat_test = df_dt_feat.iloc[split:]

class StockDataset(Dataset):
  def __init__(self, df_seq, feat_num, seq_len, target_len, df_cat):
    # SEQUENTIAL PART
    self.df_seq = df_seq.iloc[:,:-target_len]
    self.df_cat = df_cat
    self.target = df_seq.iloc[:,-target_len:]

  def __getitem__(self, index):
    return(torch.tensor(self.df_seq.iloc[index].values.reshape(seq_len,feat_num), dtype=torch.float, device=device),
           torch.tensor(self.df_cat.iloc[index], dtype=torch.long, device=device),
           torch.tensor(self.target.iloc[index], dtype=torch.float, device=device))
  
  def __len__(self):
    return(self.df_seq.shape[0])

seq_len = 30 # ข้อมูล 30 วัน
target_len = 1

bs = 64

feat_num=8
test_ds = StockDataset(df_seq_test1, feat_num, seq_len, target_len, df_dt_feat_test)
test_dl = DataLoader(test_ds, batch_size=bs, shuffle=False)

config = {'rnn_p':0., 'rnn_l':1, 'rnn_h':1000, 'seq_len':seq_len, 'rnn_input_dim':feat_num, #จำนวน feat ปัจจุบัน ให้ลอง mod ดู
          'fc_szs':[1000, 500],'fc_ps':[0.5, 0.25], 'out_sz':target_len,
          'emb_p':0.05}
cat_dict = {'day':31,'dayofweek':5,'dayofyear':366,'daysinmonth':4,'is_month_end':2,
            'is_month_start':2,'is_quarter_end':2,'is_quarter_start':2,'is_year_end':2,'is_year_start':2}
  
# lazy embedding size rule!
def emb_sz_rule(n_cat): return min(50,n_cat)


class StockPredictor(nn.Module):
  def __init__(self,config,cat_size):
    super(StockPredictor, self).__init__()
    # set parameters
    self.rnn_p = config['rnn_p']
    self.rnn_l = config['rnn_l']
    self.rnn_h = config['rnn_h']
    self.seq_len = config['seq_len']
    self.fc_szs = config['fc_szs']
    self.fc_ps = config['fc_ps']
    self.out_sz = config['out_sz']
    self.cat_size = cat_size
    self.emb_p  = config['emb_p']
    
    # embedding layers for categorical features
    self.emb_drop = nn.Dropout(self.emb_p)
    self.cat_layers = nn.ModuleList([nn.Embedding(val,emb_sz_rule(val)) for val in cat_size.values()])
    self.lin_in = sum([emb_sz_rule(val) for val in cat_size.values()]) + self.rnn_h*self.seq_len
    self.fc_szs = [self.lin_in] + self.fc_szs
    
    # recurrent layers
    self.rnn_layers = nn.LSTM(config['rnn_input_dim'], self.rnn_h, num_layers=self.rnn_l,
                             bias=True, batch_first=True, dropout=self.rnn_p)
    
    # fully connected layers
    fc_layers_list = []
    for ni, nf, p in zip(self.fc_szs[:-1], self.fc_szs[1:], self.fc_ps):
      fc_layers_list.append(nn.Linear(ni, nf))
      fc_layers_list.append(nn.ReLU(inplace=True))
      fc_layers_list.append(nn.BatchNorm1d(nf))
      fc_layers_list.append(nn.Dropout(p=p))
    self.fc_layers = nn.Sequential(*fc_layers_list)
    
    # output
    self.out = torch.nn.Linear(in_features=self.fc_szs[-1], out_features=self.out_sz)
    
  def forward(self, seq_input, cat_input):
    # cat
    cat_list = [e(cat_input[:,i]) for i,e in enumerate(self.cat_layers)]
    cat_out = torch.cat(cat_list,1)
    cat_out = self.emb_drop(cat_out)
    
    # seq
    bs = seq_input.shape[0]
    seq_out, seq_h = self.rnn_layers(seq_input)
    seq_out = seq_out.contiguous()
    seq_out = seq_out.view(bs,-1)
    
    #linear
    lin_in = torch.cat([cat_out, seq_out],1)
    res = self.fc_layers(lin_in)
    res = self.out(res)
    
    return res

PATH = './data/model.pth'

gdd.download_file_from_google_drive(file_id='1S3RI5qaM8AwOuISFl54Y93Lzt58dj5ZP',
                                    dest_path=PATH,
                                    )

seq_input,cat_input,target = next(iter(test_dl))



#model = joblib.load('./data/model.pkl')
device = torch.device(device)
model = StockPredictor(config,cat_dict)
model.load_state_dict(torch.load(PATH))
model.to(device)
model.eval()

pred_pct = model(seq_input,cat_input)

seq_input,cat_input,target = next(iter(test_dl))

# model output
pred_pct = model(seq_input,cat_input)
inv_pred_pct = pred_pct*(max_dict['pct_change']-min_dict['pct_change'])+min_dict['pct_change']
inv_true_pct = target*(max_dict['pct_change']-min_dict['pct_change'])+min_dict['pct_change']
date_test = date_all[split:]


#Streamlit display chart
st.write("""
# Changes Percentage Prediction
"""
    )
ls_inv_true_pct = pd.DataFrame(inv_true_pct.cpu().detach().numpy(),
                                              columns =['True %'])
ls_inv_pred_pct = pd.DataFrame(inv_pred_pct.cpu().detach().numpy(),
                                              columns =['Predicted %'])
chart_percent = pd.concat([ls_inv_true_pct, ls_inv_pred_pct], axis=1)
chart_percent = chart_percent.set_index(date_test)

st.write(chart_percent)
st.write("""
 Graph
""")
st.line_chart(chart_percent)





# convert to price plot
st.write("""
# Price Prediction
"""
    )
seq_input,cat_input,target = next(iter(test_dl))

# model output
pred_pct = model(seq_input,cat_input)
inv_pred_pct = pred_pct*(max_dict['pct_change']-min_dict['pct_change'])+min_dict['pct_change']

# to get real price is to multiply the predicted % with previous day close price
#edit changed seq_input[:,-1,-2] to seq_input[:,-1,-4]
prev_day_close = seq_input[:,-1,-4]*(max_dict['Adj Close']-min_dict['Adj Close'])+min_dict['Adj Close']
pred_price = (1+inv_pred_pct.view(-1,))*prev_day_close
inv_true_pct = target*(max_dict['pct_change']-min_dict['pct_change'])+min_dict['pct_change']
true_price = (1+inv_true_pct.view(-1,))*prev_day_close



#grabs the same amount of real price as pred price 
n_days = list(pred_price.shape)
n_days = n_days[0]
n_days = -n_days
tensor_Close = torch.tensor(stock_data1.iloc[n_days:,:]['Close'].values)
tensor_Close = tensor_Close.to(device)



df_actual_Close = pd.DataFrame(tensor_Close.cpu().detach().numpy(),
                                              columns =['actual price(from Close in stock_data)'])
df_true_price = pd.DataFrame(true_price.cpu().detach().numpy(),
                                              columns =['True price(calculated from %)'])
df_pred_price = pd.DataFrame(pred_price.cpu().detach().numpy(),
                                              columns =['Predicted price'])

chart_price = pd.concat([df_actual_Close, df_true_price, df_pred_price], axis=1)
chart_price = chart_price.set_index(date_test)

st.write(chart_price)

st.write("""
 Graph
""")
st.line_chart(chart_price)

st.write("SME loss of true price and predicted price")

st.write(F.mse_loss(true_price,pred_price).sqrt())

st.write("""
# Suggestion
""" )

if true_price[-1] > true_price[-2]:
    st.write("Today's predicted close price is higher than yesterday, you should buy at below average price and sell later today.")
elif true_price[-1] < true_price[-2]:
    st.write("Today's predicted close price is lower than yesterday, you should wait for the price to go down and buy later.")
else:
    st.write("Today's predicted close price is the same as yesterday, the price might be stabalizing")

#yay
