
# read input list
library(readr); library(dplyr); library(stringr)

shShares <- read_delim(file=file.path("[FILE PATH TO]/formatted_shAshare.txt"), delim = " ", na = "NA")
shShares$A股上市日期 <- as.Date(shShares$A股上市日期, format="%Y-%m-%d")
# rechieve data of the newStocks selected for the year 2016, season 1-4
# filter for stocks issued in the past year
newStocks <- shShares %>% filter(.[["A股上市日期"]] >= as.Date("2016-01-01")) %>% .[["A股代码"]]
allShAShares <- shShares %>% .[["A股代码"]]

# batch download data
# a subfolder /sina_data will be created in the given path
batch.get.stock(newStocks, c("2016"), c("1", "2", "3", "4"), "/home/jj/data/stock")
# for all sh A shares
batch.get.stock(allShAShares, c(2006:2016), c("1", "2", "3", "4"), "/home/jj/data/stock")


# Check and complete missing data
check.complete.data(newStocks, c("2016"), c("1", "2", "3", "4"), "/home/jj/data/stock")
# for all sh A shares
check.complete.data(allShAShares, c(2006:2016), c("1", "2", "3", "4"), "/home/jj/data/stock")

# read and merge selected stocks, into a subfolder
select.read.merge.by.stock(newStocks, c("2016"), c("1", "2", "3", "4"), "/home/jj/data/stock")

##########
# financials
batch.sina.financials(stocks=allShAShares[26:length(allShAShares)], yrs=c(2006:2016), statements=c("FinanceSummary","BalanceSheet", "ProfitStatement", "CashFlow"), "/home/jj/data/stock")





