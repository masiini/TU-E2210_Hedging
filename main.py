import numpy as np
import pandas as pd
from pandas.core.arrays import boolean
import pyfinance as pf
from scipy.stats import norm
import matplotlib.pyplot as plt
from pyfinance.options import BSM
from mpl_toolkits import mplot3d

eps = np.finfo(float).eps

def readData(filename):
    # Reads in the market data into a dict. 
    e  = pd.ExcelFile(filename)
    dfSheets = {}
    for indx, sheet in enumerate(e.sheet_names):
        df = e.parse(e.sheet_names[indx])
        df.columns = ["TimeToMat", *df.columns[1:-3], "Stock", "Rate", "Date"]
        dfSheets[str(indx+1) + str(sheet)] = df
    return dfSheets   

# Functions for different calculations.

def d1(strikePrice, timeTerm, stockPrice, rate, volatility):
    return (np.log(stockPrice/strikePrice) + (rate + volatility ** 2 / 2) * timeTerm) / (volatility * np.sqrt(timeTerm) + eps)

def delta(d1):
    return norm.cdf(d1, 0.0, 1.0)

def impliedVolatility(strikePrice, timeTerm, callPrice, stockPrice, rate):
    return BSM(S0=stockPrice, K=strikePrice, T=timeTerm, r=rate, sigma=0.8, kind='call').implied_vol(callPrice)
    
def vega(d1, stockPrice, timeTerm):
    return stockPrice * np.sqrt(timeTerm) * norm.pdf(d1, 0.0, 1.0)

# Functions for calculating greeks
    
def deltaHedge(data):
    strike = data.columns[1]

    # ROW: 0 = TimeToMat, 1 = Call price, 2 = Stock price, 3 = Rate
    for index, row in data.iterrows():
        volatility = impliedVolatility(strike/1000, row.iat[0]/252, row.iat[1]/1000, row.iat[2]/1000, row.iat[3]/100)
        deltaOp = delta(d1(strike/1000, row.iat[0]/252, row.iat[2]/1000, row.iat[3]/100, volatility))
        data.loc[index, ["volatility", "delta"]] = [volatility ,deltaOp]

    return data.dropna()
            
def deltaVegaHedge(dataTuple):
    dataOption, dataReplica = dataTuple[0], dataTuple[1]
    strike = dataOption.columns[1]

    # ROW: 0 = TimeToMat, 1 = Call price, 2 = Stock price, 3 = Rate
    for (indexO, rowO), (indexR, rowR) in zip(dataOption.iterrows(), dataReplica.iterrows()):
        volatilityOption = impliedVolatility(strike/1000, rowO.iat[0]/252, rowO.iat[1]/1000, rowO.iat[2]/1000, rowO.iat[3]/100)
        volatilityReplica = impliedVolatility(strike/1000, rowR.iat[0]/252, rowR.iat[1]/1000, rowR.iat[2]/1000, rowR.iat[3]/100)
        
        deltaOp = delta(d1(strike/1000, rowO.iat[0]/252, rowO.iat[2]/1000, rowO.iat[3]/100, volatilityOption))
        deltaRep = delta(d1(strike/1000, rowR.iat[0]/252, rowR.iat[2]/1000, rowR.iat[3]/100, volatilityReplica))

        vegaOp = vega(d1(strike/1000, rowO.iat[0]/252, rowO.iat[2]/1000, rowO.iat[3]/100, volatilityOption), rowO.iat[2]/1000, rowO.iat[0]/252)
        vegaRep = vega(d1(strike/1000, rowR.iat[0]/252, rowR.iat[2]/1000, rowR.iat[3]/100, volatilityReplica), rowR.iat[2]/1000, rowR.iat[0]/252)

        n = vegaOp / vegaRep
        alfa = deltaOp - n * deltaRep

        dataOption.loc[indexO, ["volOption", "volRep", "callRep", "deltaOption", "deltaReplica", "vegaOption", "vegaReplica", "alfa", "n"]] = \
                                [volatilityOption, volatilityReplica, rowR.iat[1], deltaOp, deltaRep, vegaOp, vegaRep, alfa, n]

    return dataOption.dropna(thresh=14)
    
# Portfolio functions

def deltaPortfolio(data, hedgeFreq):
    # Indexes of 2 = Stock price, 5 = Delta, 6 = A, 7 = RE
    if not data.empty:
        data = data.assign(A=data.iloc[::hedgeFreq, 1].diff() - data.iloc[::hedgeFreq, 2].diff().mul(data.iloc[::hedgeFreq, 5].shift(periods=1, fill_value=0)),\
            REHedgeCost=data.iloc[::hedgeFreq, 5].diff().mul(data.iloc[::hedgeFreq, 2]) )
        data.fillna(0, inplace=True)
        data.iat[0, 7], data.iat[-1,7] = - data.iat[0,1] + data.iat[0, 2] * data.iat[0, 5], \
                                        data.iat[-1,2] - data.iat[-1,2] * data.iat[-1,5] if data.iat[-1,2] - data.columns[1] > 0 else - data.iat[-1,2] * data.iat[-1,5]
        
        a = data.iloc[::hedgeFreq, 6].pow(2).sum() / data.iloc[::hedgeFreq, 6].count()
        # Cost is shorted position hedging minus the cost of buying back stocks on the last day 
        totalPL =  data.iloc[:, 7].sum()
    else:
        a, totalPL = np.nan, np.nan

    return data, (a, totalPL)
    
def deltaVegaPortfolio(data, hedgeFreq):
    # Indexes of 2 = Stock price, 5 = Call price of replica, 12 = Alfa, 13 = n, 14 = A
    if not data.empty: 
        data = data.assign(A=(data.iloc[::hedgeFreq, 1].div(1000).diff() * (-1)) + (data.iloc[::hedgeFreq, 2].mul(data.iloc[::hedgeFreq, 12]).div(1000).diff()) + data.iloc[::hedgeFreq, 7].mul(data.iloc[::hedgeFreq, 13]).div(1000).diff(), \
                            hedgeCost=(data.iloc[::hedgeFreq,2].mul(data.iloc[::hedgeFreq, 12]).diff() + data.iloc[::hedgeFreq, 8].mul(data.iloc[::hedgeFreq, 13]).diff()) * (-1))
        data.fillna(0, inplace=True)
        # Short call - stocks purchased - calls puchased - shorted option payoff + sold stock position + sold replicated option
        data.iat[0,15] = data.iat[0,1] - (data.iat[0,2] * data.iat[0,12] + data.iat[0,8] * data.iat[0,13])
        data.iat[-1, 15] = - data.iat[-1,2] + (data.iat[-1,2] * data.iat[-1,12] + data.iat[-1,8] * data.iat[-1,13]) if data.iat[-1,2] - data.columns[1] > 0 else (data.iat[-1,2] * data.iat[-1,12] + data.iat[-1,8] * data.iat[-1,13])

        a = data.iloc[::hedgeFreq, 14].pow(2).sum() / data.iloc[::hedgeFreq, 14].count()
        totalPL = data.iloc[:, 15].sum()
    else:
        a, totalPL = np.nan, np.nan

    return data, (a, totalPL)


def main():
    # Functions for tidying up and enabling longer maturities in the worksheet data
    def deltaData(data, timeToMat, strikePrice):
        return data.loc[data["TimeToMat"] <= timeToMat].iloc[:,[0,data.columns.get_loc(strikePrice),-3,-2]].dropna().reset_index(drop=True)

    def deltaVegaData(dataOption, dataReplica, timeToMat, strikePrice):
        dateToMatch, matDate = dataOption.loc[dataOption["TimeToMat"] == timeToMat, "Date"].iat[0], dataOption.loc[dataOption["TimeToMat"] == 1, "Date"].iat[0]
        replicaTimeToMat, replicaMatDate = dataReplica.loc[dataReplica["Date"] == dateToMatch, "TimeToMat"].iat[0], dataReplica.loc[dataReplica["Date"] == matDate, "TimeToMat"].iat[0]
        option = dataOption.loc[dataOption["TimeToMat"] <= timeToMat].iloc[:,[0,dataOption.columns.get_loc(strikePrice),-3,-2,-1]].dropna().reset_index(drop=True)
        replica = dataReplica.loc[(dataReplica["TimeToMat"] <= replicaTimeToMat) & (dataReplica["TimeToMat"] >= replicaMatDate)].iloc[:,[0,dataReplica.columns.get_loc(strikePrice),-3,-2,-1]].dropna().reset_index(drop=True)
        return (option, replica)

    dfDict = readData("isx2010C.xls")

    # Uncomment figures for analysis. No reason for not using plt.savefigure() for saving.

    # 2.1 delta hedging a single option

    # Figure for deltas (OK)
    # for strikePrice in dfDict["1isx15012010C"].columns[15:-3:4].values:
    #     deltas = deltaHedge(deltaData(dfDict["1isx15012010C"], 30, strikePrice))
    #     plt.plot(deltas.index.array + 1, deltas.iloc[:,5], label=strikePrice)
    # plt.legend()
    # plt.title("Deltas for different strike prices")
    # plt.ylabel("Delta")
    # plt.xlabel("Days")
    # plt.show()


    # Figure for Volatilities (OK)
    # for strikePrice in dfDict["1isx15012010C"].columns[15:-3:4].values:
    #     deltas = deltaHedge(deltaData(dfDict["1isx15012010C"], 30, strikePrice))
    #     plt.plot(deltas.index.array + 1, deltas.iloc[:,4], label=strikePrice)
    # plt.legend()
    # plt.title("Volatilities for different strike prices")
    # plt.ylabel("Volatility")
    # plt.xlabel("Days")
    # plt.show()


    # Figure for A (OK)
    # ax = plt.axes(projection="3d")
    # for strikePrice in dfDict["1isx15012010C"].columns[15:-3:4].values:
    #     deltas = deltaHedge(deltaData(dfDict["1isx15012010C"], 30, strikePrice))
    #     for rehedgeFreq in range(1,15):
    #         data, stats = deltaPortfolio(deltas, rehedgeFreq)
    #         ax.plot(strikePrice, rehedgeFreq, stats[0], 'ro')
    # ax.set_xlabel("Strike price")
    # ax.set_ylabel("Rehedging frequency")
    # ax.set_title("Mean squared error with 30 day TTM")
    # plt.show()

    # Figure for cost (OK)
    # deltas = deltaHedge(deltaData(dfDict['7isx20082010C'], 30, 480))
    # for rehedgeFreq in range(1,8):
    #     data, stats = deltaPortfolio(deltas, rehedgeFreq)
    #     if not data.empty:
    #         plt.plot(data.index.array + 1, data.iloc[:, 7].cumsum(), label=rehedgeFreq)
    #         if stats[1] > 0:
    #             print(f"{rehedgeFreq}: {stats[1]}")
    # plt.xlabel("Days")
    # plt.ylabel("Cost")
    # plt.title("485 strike cumulative cost with different re-hedging frequencies")
    # plt.legend()
    # plt.grid()
    # plt.show()


    # 2.2 delta-vega hedging a single option

    # Figure for vega (OK)
    # for strikePrice in dfDict['5isx15102010C'].columns[13:-4:4].values:
    #     deltavegas = deltaVegaHedge(deltaVegaData(dfDict['5isx15102010C'], dfDict['4isx19112010C'], 30, strikePrice))
    #     plt.plot(deltavegas.index.array + 1, deltavegas.iloc[:,10].to_numpy(), label=strikePrice)
    # plt.xlabel("Days")
    # plt.ylabel("Vegas")
    # plt.title("Vegas for different strike prices")
    # plt.legend()
    # plt.show()


    # Figure for volatilities (Add labels)
    # for strikePrice in dfDict['8isx16072010C'].columns[12:-4:4].values:
    #     deltavegas = deltaVegaHedge(deltaVegaData(dfDict['8isx16072010C'], dfDict['7isx20082010C'], 30, strikePrice))
    #     plt.plot(deltavegas.index.array + 1, deltavegas.iloc[:, 5].to_numpy(), label=strikePrice)
    # plt.xlabel("Days")
    # plt.ylabel("Volatility")
    # plt.title("Volatilities with different strike prices")
    # plt.legend()
    # plt.show()

    # Figure for A (OK)
    # ax = plt.axes(projection="3d")
    # for strikePrice in dfDict['8isx16072010C'].columns[18:-10:2].values:
    #     deltavegas = deltaVegaHedge(deltaVegaData(dfDict['8isx16072010C'], dfDict['7isx20082010C'], 30, strikePrice))
    #     for rehedgeFreq in range(1,15):
    #         data, stats = deltaVegaPortfolio(deltavegas, rehedgeFreq)
    #         ax.plot(strikePrice, rehedgeFreq, stats[0], 'ro')
    # ax.set_xlabel("Strike price")
    # ax.set_ylabel("Rehedging frequency")
    # ax.set_title("Mean squared error with 30 day TTM")
    # plt.show()

    # Figure for cost

    # deltaVegas = deltaVegaHedge(deltaVegaData(dfDict['8isx16072010C'], dfDict['7isx20082010C'], 30, 500))
    # for rehedgeFreq in range(1,8):
    #     data, stats = deltaVegaPortfolio(deltaVegas, rehedgeFreq)
    #     if not data.empty:
    #         plt.plot(data.index.array + 1, data.iloc[:, 15].cumsum(), label=rehedgeFreq)
    #         if stats[1] > 0:
    #             print(f"{rehedgeFreq}: {stats[1]}")
    # plt.xlabel("Days")
    # plt.ylabel("Cost")
    # plt.title("500 strike cumulative cost with different re-hedging frequencies")
    # plt.legend()
    # plt.grid()
    # plt.show()

    #3.1 delta hedging a portfolio of options
    
    # Bull spread
    # Version 1
    # longDelta = deltaHedge(deltaData(dfDict["5isx15102010C"], 40, 485))
    # shortDelta = deltaHedge(deltaData(dfDict["5isx15102010C"], 40, 525))

    # Figures for deltas (OK)
    # plt.plot(longDelta.index.array + 1, longDelta.iloc[:,5], label='Long 485')
    # plt.plot(shortDelta.index.array + 1, shortDelta.iloc[:,5], label='Short 525')
    # plt.legend()
    # plt.title("Deltas for bull spread with strikes 485 & 525")
    # plt.ylabel("Delta")
    # plt.xlabel("Days")
    # plt.show()

    # Figures for A (OK)
    # ax = plt.axes(projection="3d")
    # for rehedgeFreq in range(1,15):
    #     data, stats = deltaPortfolio(longDelta, rehedgeFreq)
    #     ax.plot(485, rehedgeFreq, stats[0], 'ro')
    #     data, stats = deltaPortfolio(shortDelta, rehedgeFreq)
    #     ax.plot(525, rehedgeFreq, stats[0], 'bo')        
    # ax.set_xlabel("Strike price")
    # ax.set_ylabel("Rehedging frequency")
    # ax.set_title("Mean squared error with 30 day TTM")
    # plt.show()

    # Version 2
    # longDelta = deltaHedge(deltaData(dfDict["9isx18062010C"], 60, 520))

    # shortDelta = deltaHedge(deltaData(dfDict["9isx18062010C"], 60, 560))

    # Figures for deltas (OK)
    # plt.plot(longDelta.index.array + 1, longDelta.iloc[:,5], label='Long 520')
    # plt.plot(shortDelta.index.array + 1, shortDelta.iloc[:,5], label='Short 560')
    # plt.legend()
    # plt.title("Deltas for bull spread with strikes 520 & 560")
    # plt.ylabel("Delta")
    # plt.xlabel("Days")
    # plt.show()

    # Figures for A (OK)
    # ax = plt.axes(projection="3d")
    # for rehedgeFreq in range(1,15):
    #     data, stats = deltaPortfolio(longDelta, rehedgeFreq)
    #     ax.plot(520, rehedgeFreq, stats[0], 'ro')
    #     data, stats = deltaPortfolio(shortDelta, rehedgeFreq)
    #     ax.plot(560, rehedgeFreq, stats[0], 'bo')        
    # ax.set_xlabel("Strike price")
    # ax.set_ylabel("Rehedging frequency")
    # ax.set_title("Mean squared error with 30 day TTM")
    # plt.show()


    # Butterfly spread

    # longDelta1 = deltaHedge(deltaData(dfDict["9isx18062010C"], 30, 480))
    # longDelta2 = deltaHedge(deltaData(dfDict["9isx18062010C"], 30, 520))
    # shortDelta = deltaHedge(deltaData(dfDict["9isx18062010C"], 30, 500))


    # 3.2 delta-vega hedging a portfolio of options

    # Bull Spread 

    # Figure for vegas (OK)
    # longDeltaVega = deltaVegaHedge(deltaVegaData(dfDict['8isx16072010C'], dfDict['7isx20082010C'], 30, 470))
    # shortDeltaVega = deltaVegaHedge(deltaVegaData(dfDict['8isx16072010C'], dfDict['7isx20082010C'], 30, 500))

    # plt.plot(longDeltaVega.index.array + 1, longDeltaVega.iloc[:,10].to_numpy(), label=470)
    # plt.plot(shortDeltaVega.index.array + 1, shortDeltaVega.iloc[:,10].to_numpy(), label=500)
    # plt.xlabel("Days")
    # plt.ylabel("Vega")
    # plt.title("Vegas for bull spread strategy")
    # plt.legend()
    # plt.show()

    # Figure for A
    # ax = plt.axes(projection="3d")
    # for rehedgeFreq in range(1,15):
    #     data, stats = deltaVegaPortfolio(longDeltaVega, rehedgeFreq)
    #     ax.plot(470, rehedgeFreq, stats[0], 'bo')
    #     data, stats = deltaVegaPortfolio(shortDeltaVega, rehedgeFreq)
    #     ax.plot(500, rehedgeFreq, stats[0], 'ro')        
    # ax.set_xlabel("Strike price")
    # ax.set_ylabel("Rehedging frequency")
    # ax.set_title("Mean squared error with 30 day TTM")
    # plt.legend()
    # plt.show()


if __name__ == "__main__":
    main()