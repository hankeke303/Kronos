import numpy as np
import argparse
from torch.cuda.amp import autocast
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import pandas as pd
from operator import itemgetter
parser=argparse.ArgumentParser()

parser.add_argument('--chip',        default=5,     type=int)
parser.add_argument('--inter',      default='3',     choices=['2','3','5','10','15'])
parser.add_argument('--thres_step',      default=0.2, type=float)
args = parser.parse_args()

dfs = pd.DataFrame(columns=[
    'modelname', 'buythreshold', 'sellthreshold',
    'finalmoney', 'maxhuiche', 'buypoint',
    'chip', 'buyacc'
])
def mapsubscript(alldict,moving_stock):
    return np.array(itemgetter(*moving_stock)(alldict))

alldata = np.load('dtprice_stdict.npy', allow_pickle=True).item()
allsignals=np.load('mapsignals.npy',allow_pickle=True).item()
stock_real_dict = np.load('dtstockdict.npy', allow_pickle=True).item()

startime =  next(iter(allsignals))
end_time = next(reversed(allsignals))
# indexss = [6, 7, 8,9,15,17,19]
# alldata = np.delete(alldata, indexss,axis=3)
nowdate = np.array(sorted(list(allsignals.keys())))
buythreshold = 0.5
for ups in range(20):
    chip = args.chip
    stockdict = np.sort(np.load('stockhyalldict.npy', allow_pickle=True))

    indexdict = dict(zip(stockdict,range(len(stockdict))))
    stock2idx = {s: i for i, s in enumerate(stockdict)}
    buythreshold = buythreshold + args.thres_step
    allmoney = 10000000
    exchange_fee = 0.0015
    sellthrehsold = -10  # 最好参数 1.3 0.3 -0.05  126w
    buypoint = 0
    sellpoint = 0
    allbuyacc = 0
    premoney = allmoney
    # datetime.datetime.strptime('2020-11-13 09:00:00', "%Y-%m-%d %H:%M:%S")-datetime.datetime.strptime(allbuydate[sell], "%Y-%m-%d %H:%M:%S")
    allmappingdict = dict(zip(stockdict, np.arange(len(stockdict))))
    df = pd.DataFrame(columns=('times','stock_name','money','operation','close','earnmoney'))
    capital=[]
    allchip=[]
    realchip=chip
    allbuylist=np.zeros(len(stockdict))
    allbuyclose=np.zeros(len(stockdict))
    allbuynum=np.zeros(len(stockdict))
    buynowamount=np.zeros(len(stockdict))
    allbuydate=np.zeros(len(stockdict)).astype(np.str_)
    handmoney=allmoney
    stock_money=np.zeros(len(stockdict))
    ifbuy=0
    yichangnum=0
    allyichang=[]
    allmax5=[]
    allmean=[]
    already_buy=[]
    for inddt,date in enumerate(nowdate):
        data = alldata[date]
        moving_name = np.array(stock_real_dict[date])
        movemappingdict = dict(zip(moving_name, np.arange(len(moving_name))))
        #test_loader = DataLoader(datas, batch_size=64, shuffle=False, pin_memory=True)
        stocks_today = stock_real_dict[date]  # 当日 N 个股票名
        today_signals = allsignals[date]
        result = today_signals

        nowtime_index = np.where(nowdate == date)[0]
        alldownbuy = []
        allindexsell = []
        # for numss in range(10):
        print(np.max(result))
        #    allyichang.append(result[movemappingdict['300162']])
        # allmax5.append(np.mean(np.sort(result)[-5:]))

        # allmaxmean.append(np.mean(np.sort(result)[-10:]))
        # allminmean.append(np.mean(np.sort(result)[:10]))
        allmean.append(np.mean(result))
        movingindexbuy = np.where(result >= buythreshold)[0]
        try:
            allindexbuy = mapsubscript(allmappingdict, moving_name[movingindexbuy])  # 将动态的股票名索引到静态的下标
        except:
            allindexbuy = np.array([])
        try:
            if len(allindexbuy):
                pass
        except:
            allindexbuy = np.array([allindexbuy])
        try:
            for inbuy in allindexbuy:
                if data[movemappingdict[stockdict[inbuy]],-1]>0:
                    allindexbuy = np.delete(allindexbuy, np.where(allindexbuy == inbuy)[0][0])
                elif data[movemappingdict[stockdict[inbuy]],0] <= 5:
                    allindexbuy = np.delete(allindexbuy, np.where(allindexbuy == inbuy)[0][0])
        except:
            pass
        # np.array(itemgetter(*moving_name[movingindexbuy])(allmappingdict))
        already_buy = np.where(allbuylist == 1)[0]
        for already in already_buy:
            try:
                if data[movemappingdict[stockdict[already]],0]:
                    nowbuy_index = np.where(nowdate == allbuydate[already])[0]
                    if result[movemappingdict[stockdict[already]]] <= sellthrehsold:
                        allindexsell.append(already)
                    elif (nowtime_index - nowbuy_index) >=int(args.inter):
                        allindexsell.append(already)
            except:
                continue
            # 规则止损
        for already in already_buy:
            try:
                if data[movemappingdict[stockdict[already]], 0]:
                    buynowamount[already] = data[movemappingdict[stockdict[already]], 0] * allbuynum[
                        already]
            except:
                buynowamount[already] = allbuyclose[already] * allbuynum[already]

        allindexsell = np.unique(np.hstack((np.array(allindexsell), np.array(alldownbuy)))).astype(int)

        # zeroerrorsell = np.where(data[allindexsell, -1, 6, 0] == 0)[0]
        '''for alldown in alldownbuy:
            try:
                allindexbuy = np.delete(allindexbuy,np.where(allindexbuy==alldown))
            except:
                pass'''
        # allindexsell =np.delete(allindexsell, zeroerrorsell)

        # allmoney=handmoney+stockmoney newtime compute allmoney
        allmoney = handmoney
        for already in already_buy:
            allmoney = allmoney + buynowamount[already]  # 滑点问题  报错
        buynowamount[already_buy] = 0

        for already in already_buy:
            try:
                if data[movemappingdict[stockdict[already]], 0]:
                    stock_money[already] = data[movemappingdict[stockdict[already]], 0] * allbuynum[
                        already]
            except:
                continue

        realsell = np.intersect1d(allindexsell, already_buy)  # 获取实际要卖出的股票
        for sell in realsell:
            # money_ratio=(data[sell,-1,6,0]-allbuyclose[sell])/allbuyclose[sell]
            earn_money = stock_money[sell] - allbuyclose[sell] * allbuynum[sell]
            # handmoney=handmoney+stock_money[sell]*(1-exchange_fee)
            allmoney = allmoney - stock_money[sell] * (exchange_fee)
            df.loc[len(df)] = {
                'times': date,
                'stock_name': stockdict[sell],
                'money': allmoney,
                'operation': 2,
                'close': data[movemappingdict[stockdict[sell]], 0],
                'earnmoney': earn_money
            }
            # ??????

            if earn_money - stock_money[sell] * (exchange_fee * 2) > 0:
                allbuyacc = allbuyacc + 1
            print('sell', date, stockdict[sell], 'buymoney', allbuyclose[sell], 'sellmoney',
                  data[movemappingdict[stockdict[sell]], 0],
                  'earnratio', earn_money / (allbuyclose[sell] * allbuynum[sell]))
            stock_money[sell] = 0
            allbuynum[sell] = 0
            allbuyclose[sell] = 0
            allbuylist[sell] = 0
            allbuydate[sell] = '0'
            realchip = realchip + 1
            sellpoint = sellpoint + 1

        already_buy = np.where(allbuylist == 1)[0]
        realbuy = np.setdiff1d(allindexbuy, already_buy)  # 获取实际要买入的股票

        if len(realbuy) > realchip:
            realbuy = np.vstack(
                [realbuy, result[mapsubscript(movemappingdict, np.array(stockdict[realbuy]))]])  # 按reward大小选股票
            realbuy = realbuy.T[np.lexsort(-realbuy)].T
            allrealbuy = realbuy[0].copy().astype(int)
            realbuy = realbuy[0][:realchip].astype(int)  # 如果筹码不足，按顺序取，后面可以尝试调整按reward大小取
            print('筹码不足，无法购买', stockdict[allrealbuy[realchip:]])

        for already in already_buy:
            try:
                if data[movemappingdict[stockdict[already]],0]:
                    buynowamount[already] = data[movemappingdict[stockdict[already]],0] * allbuynum[
                        already]
            except:
                buynowamount[already] = allbuyclose[already] * allbuynum[already]
        # 卖完之后更新handmoney
        try:
            handmoney = allmoney - np.sum(buynowamount[already_buy])
        except:
            handmoney = allmoney

        print('sellall:', handmoney)
        if realchip != 0:
            buymoney = np.floor(handmoney * (1 - exchange_fee) / realchip)
        else:
            buymoney = buymoney
        # buynum=np.floor(buymoney/data[realbuy, -1, 6, 0]/100)*100
        try:
            allbuynum[realbuy] = np.floor(buymoney / data[
                mapsubscript(movemappingdict, np.array(stockdict[realbuy])), 0] / 100) * 100
            stock_money[realbuy] = allbuynum[realbuy] * data[
                mapsubscript(movemappingdict, np.array(stockdict[realbuy])),0]
            allmoney = allmoney - np.sum(stock_money[realbuy]) * exchange_fee
            allbuylist[realbuy] = 1
            realchip = realchip - len(realbuy)
            allbuyclose[realbuy] = data[[mapsubscript(movemappingdict, np.array(stockdict[realbuy]))], 0]
            allbuydate[realbuy] = date
        except:
            pass
        for buys in realbuy:
            buypoint = buypoint + 1
            df.loc[len(df)] = {'times': allbuydate[buys],
                               'stock_name': stockdict[buys],
                               'money': allmoney,
                               'operation': 1,
                               'close': allbuyclose[buys],
                               'earnmoney': 0}



        already_buy = np.where(allbuylist == 1)[0]
        for already in already_buy:
            try:
                if data[movemappingdict[stockdict[already]], 0]:
                    buynowamount[already] = data[movemappingdict[stockdict[already]],0] * allbuynum[
                        already]
            except:
                buynowamount[already] = allbuyclose[already] * allbuynum[already]

        try:
            handmoney = allmoney - np.sum(buynowamount[already_buy])
        except:
            handmoney = allmoney

        print(date, 'buylist:')
        allchip.append(chip - realchip)
        print('nowchip:', realchip, allmoney, handmoney, 'buypoint:', buypoint, 'sellpoint:', sellpoint)
        capital.append(allmoney)
        buynowamount[already_buy] = 0
    if sellpoint <= 80:
        break
    huiche = []
    maxmoney = 10000000
    for num in capital:
        huiche.append(min(num - maxmoney, 0) / abs(maxmoney))
        if num >= maxmoney:
            maxmoney = num
    huiche = np.array((huiche))
    maxhuiche = -np.min(huiche)
    dfs.loc[len(dfs)] ={'modelname': 'kronos',
                        'buythreshold': buythreshold,
                        'sellthreshold': sellthrehsold,
                         'finalmoney': allmoney,
                        'maxhuiche':maxhuiche,
                        'buypoint':buypoint,
                        'chip':chip,
                        'buyacc':allbuyacc/sellpoint}
    dfs.to_csv('rizhikronos'+args.inter+'.csv')

