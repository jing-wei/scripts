
#########################################

##### 新浪实时数据接口 day trade information
# example
# http://hq.sinajs.cn/list=sh601857



library(RCurl)

# test example
# stock.code <- "sh600000"

get.stock.info <- function (stock.code) {
        
        # get the stock information using http/js interface
        url <- paste0('http://hq.sinajs.cn/list=', stock.code)
        #stock.info <- strsplit(getURL(url, .encoding='GBK'), ',')[[1]]
        stock.info <- strsplit(getURL(url, .encoding='utf-8'), ',')[[1]]
        # get the stock symbol
        stock.symbol <- stock.info[1]
        #loc <- gregexpr(pattern='\"', stock.symbol)[[1]][1]
        #stock.info[1] <- substr(stock.symbol, loc+1, nchar(stock.symbol))
        loc <- gregexpr(pattern='=\"', stock.symbol)[[1]][1]
        stock.info[1] <- substr(stock.symbol, 12, loc-1)
        
        stock.info <- stock.info[1:(length(stock.info)-1)]
        # convert a character string to a data.frame
        stock.data <- data.frame(V1=0)
        for (i in 1:length(stock.info)) {
                stock.data[[i]] <- stock.info[i]
        }
        
        # convert string to float, except for the symbol, date, and time
        stock.data[2:(length(stock.info)-2)] <- as.numeric(stock.data[2:(length(stock.info)-2)])
        
        # return the stock data 
        stock.data
}
# input: index.code: sh000001 (上证指数) sz399001 (深圳成指)
# output: a data.frame containing the index information
get.index.info <- function(index.code) {
        # get the index information using the http/js interface
        url <- paste0('http://hq.sinajs.cn/list=s_', index.code)
        index.info <- strsplit(getURL(url, .encoding='GBK'), ',')[[1]]
        
        # get the index symbol
        index.symbol <- index.info[1]
        #loc <- gregexpr(pattern='\"', index.symbol)[[1]][1]
        #index.info[1] <- substr(index.symbol, loc+1, nchar(index.symbol))
        loc <- gregexpr(pattern='=\"', stock.symbol)[[1]][1]
        index.info[1] <- substr(index.symbol, 12, loc-1)
        
        # get the index volume of transaction
        index.volume <- index.info[length(index.info)]
        loc <- gregexpr(pattern='\"', index.volume)[[1]][1]
        index.info[length(index.info)] <- substr(index.volume, 1, loc-1)
        
        # convert a character string to a data.frame
        index.data <- data.frame(V1=0)
        for (i in 1:length(index.info)) {
                index.data[[i]] <- index.info[i]
        }
        
        # convert string to float, except for the symbol
        index.data[2:length(index.info)] <- as.numeric(index.data[2:length(index.info)])
        
        # return the index data
        index.data
}

##############################

#get.stock.info("sh601872")

stock_variables <- c("stockCode", "todayOpen", "yesterdayClose", "currentPrice", "todayHighest", "todayLowest", 
                     "buyingPrice", "sellingPrice", "stockExchanged", "sales", "buyer1Amount", "buyer1Price", 
                     "buyer2Amount", "buyer2Price", "buyer3Amount", "buyer3Price", "buyer4Amount", "buyer4Price", 
                     "buyer5Amount", "buyer5Price", "seller1Amount", "seller1Price", "seller2Amount", "seller2Price", 
                     "seller3Amount", "seller3Price", "seller4Amount", "seller4Price", "seller5Amount", "seller5Price", 
                     "date", "time")

################# index variable names
index_variables <- c("indexCode", "currentIndex", "currentPrice", "change%", "handsExchanged(100)", "amountExhcnaged(10000)")

# give column names
#colnames(stock.data) <- stock_variables
# add column names
#colnames(index.data) <- index_variables
#################################################################################################



#################################################################################################
#################################################################################################
# grab history data from sina
# http://vip.stock.finance.sina.com.cn/corp/go.php/vMS_FuQuanMarketHistory/stockid/603843.phtml?year=2016&jidu=4
library(XML); library(RCurl); library(rjson); library(stringr) 



# functions to clean scrubbed data frame
format.data.frame <- function(df) {
        nms <- c("日期", "开盘价", "最高价", "收盘价", "最低价", "交易量(股)", "交易金额(元)", "复权因子")
        dt <- as.matrix(df[-c(1,2), ])
        df_out<- data.frame(dt)
        colnames(df_out) <- nms
        return(df_out)
}

# to write get.sina.stock.history function JX 20161230
write.sina.stock.history <- function(stockid, yr, season, data_folder) {
        require(XML); require(readr); require(RCurl)
        if(!require(XML)) {
                install.packages("XML")
        }
        if(!require(readr)) {
                install.packages("readr")
        }
        # url in parts
        u1 <- "http://vip.stock.finance.sina.com.cn/corp/go.php/vMS_FuQuanMarketHistory/stockid/"
        u2 <- ".phtml?year="
        u3 <- "&jidu="
        u <- paste0(u1, stockid, u2, yr, u3, season)
        tables <- readHTMLTable(u)
        temp <- format.data.frame(tables[["FundHoldSharesTable"]])
        
        # write file in /sina_data folder in wk folder
        output_dir <- file.path(data_folder, "sina_data")
        if(!dir.exists(output_dir)) {
                dir.create(output_dir, showWarnings = TRUE, recursive = FALSE, mode = "0777")
        }
        write_csv(temp, path=file.path(output_dir, paste0(stockid, "_", yr, "_", season, ".csv")))
}

# get 
get.sina.stock.history <- function(stockid, yr, season) {
        if(!require(XML)) {
                install.packages("XML")
        }
        if(!require(readr)) {
                install.packages("readr")
        }
        # url in parts
        u1 <- "http://vip.stock.finance.sina.com.cn/corp/go.php/vMS_FuQuanMarketHistory/stockid/"
        u2 <- ".phtml?year="
        u3 <- "&jidu="
        u <- paste0(u1, stockid, u2, yr, u3, season)
        tables <- readHTMLTable(u)
        temp <- format.data.frame(tables[["FundHoldSharesTable"]])
        #str(tables)
        return(temp)
        # write file in /sina_data folder in wk folder
        #write_csv(temp, path=file.path(getwd(), "sina_data", paste0(stockid, "_", yr, "_", season, ".csv")))
}

# error handler
sina.error.handler <- function(stock, yr, season, data_folder) {
        tryCatch(
                {
                        message(paste("Trying", stock, yr, season))
                        write.sina.stock.history(stock, yr, season, data_folder)
                }, 
                # give a second try after 5s, in case of error(fail to grab data)
                error=function(stock2, yr2, season2) {
                        stock2 <- stock; yr2 <- yr; season2 <- season
                        Sys.sleep(5)
                        current_file <- paste(stock2, yr2, season2)
                        message(paste("Trying again", current_file))
                        sina.error.handler(stock2, yr2, season, data_folder)
                }
        )
}


# function to scrub and write data in batch
batch.get.stock <- function(stocks, yrs, seasons, data_folder) {
        for(stock in stocks) {
                for(yr in yrs) {
                        for(season in seasons) {
                                #
                                tryCatch(
                                        {
                                                message(paste("This is first try of", stock, yr, season))
                                                write.sina.stock.history(stock, yr, season, data_folder)
                                        }, 
                                        # give a second try after 5s, in case of error(fail to grab data)
                                        error=function(stock2, yr2, season2) {
                                                stock2 <- stock; yr2 <- yr; season2 <- season
                                                Sys.sleep(5)
                                                current_file <- paste(stock2, yr2, season2)
                                                message(paste("This is second try of", current_file))
                                                sina.error.handler(stock2, yr2, season, data_folder)
                                        }
                                )
                        }
                }
        }
}



# function to check completeness of data, and redownload if missing
# Some data may be missing with empty files naming like Error in UseMethod("xmlNamespaceDefinitions")/ no applicable method for 'xmlNamespaceDefinitions' applied to an object of class "NULL"?_2016_4.csv
# Fix here: check whether files with the right names exist for each of the input stocks, regrab if not
check.complete.data <- function(stocks, yrs, seasons, data_folder) {
        # stocks: list of stocks. list of integer stock codes
        # yrs: list of years. c("2016", "2015", ...)
        # seasons: list of seasons. c("1", "2", ...)
        # data_folder: name of folder holding downloaded data in working folder, or complete folder path in mac
        # generate filenames that should exist in folder
        require(dplyr)
        require(XML)
        require(stringr)
        complete.filenames <- vector()
        i <- 1
        for(stock in stocks) {
                for(yr in yrs) {
                        for(season in seasons) {
                                tmp <- paste0(stock, "_", yr, "_", season, ".csv")
                                complete.filenames[i] <- tmp
                                i <- i + 1
                        }
                }
        }
        # check which files are missing in the folder
        real.filenames <- vector()
        if(grepl("/", data_folder)) {
                real.filenames <- list.files(data_folder)
        } else {
                real.filenames <- list.files(file.path(getwd(), data_folder))
        }
        # missing files
        missing.files <- complete.filenames [! complete.filenames %in% real.filenames]
        if(length(missing.files) == 0) {print("Data complete!")}
        # download again
        if(length(missing.files) > 0) {
                for(file in missing.files) {
                        # extract stock, yr, season
                        split.filename <- str_split(file, pattern="[.]|_")[[1]]
                        missingStock <- split.filename[1]
                        missingYr <- split.filename[2]
                        missingSeason <- split.filename[3]
                        # download again
                        tryCatch(
                                {
                                        message("This is first try")
                                        write.sina.stock.history(missingStock, missingYr, missingSeason)
                                }, 
                                # give a second try after 5s, in case of error(fail to grab data)
                                error=function(missingStock, missingYr, missingSeason) {
                                        Sys.sleep(5)
                                        message(paste("This is second try of", missingStock, missingYr, missingSeason))
                                        write.sina.stock.history(missingStock, missingYr, missingSeason)
                                }
                                
                        )
                }
        }
        # check again, download again if missing
        real.filenames2 <- vector()
        if(grepl("/", data_folder)) {
                real.filenames2 <- list.files(data_folder)
        } else {
                real.filenames2 <- list.files(file.path(getwd(), data_folder))
        }
        # check missing files again and print out if any
        missing.files2 <- complete.filenames [! complete.filenames %in% real.filenames2]
        if(length(missing.files2) > 0) {
                message(paste("The following stock data are still missing:\n", missing.files2))
        }
}

# function to read written data, and format
# read and merge data by stocks
select.read.merge.by.stock <- function(stocks, yrs, seasons, data_folder) {
        require(dplyr)
        require(readr)
        # stocks: list of stocks to extract
        # yrs: list of years
        # list of seasons
        # data_folder: path to folder where data are stored, or folder name in working folder
        
        # folder to read data
        data_path <- vector()
        if(grepl("/", data_folder)) {
                data_path <- data_folder
        } else {
                data_path <- file.path(getwd(), data_folder)
        }
        # directory for output 
        output_dir <- file.path(data_path, Sys.Date())
        if(!dir.exists(output_dir)) {
                dir.create(output_dir, showWarnings = TRUE, recursive = FALSE, mode = "0777")
        }
        
        # for every input stock
        for(stock in stocks) {
                # files to extract for each stock
                extract.filenames <- vector()
                i <- 1
                for(yr in yrs) {
                        for(season in seasons) {
                                tmp <- paste0(stock, "_", yr, "_", season, ".csv")
                                extract.filenames[i] <- tmp
                                i <- i + 1
                        }
                }
                # read and merge files for each stock
                extract.files.path <- file.path(data_path, extract.filenames)
                data.per.stock <- do.call(rbind, lapply(extract.files.path, read_csv)) %>% as.data.frame()
                # write data
                write_csv(data.per.stock, path=file.path(output_dir, paste0(stock, ".csv")))
        }
}



#################################################################################################
#################################################################################################
# grab financial statements from sina
# example urls
# http://money.finance.sina.com.cn/corp/go.php/vFD_FinanceSummary/stockid/603843/displaytype/4.phtml
# http://money.finance.sina.com.cn/corp/go.php/vFD_BalanceSheet/stockid/603843/ctrl/2016/displaytype/4.phtml
# http://money.finance.sina.com.cn/corp/go.php/vFD_ProfitStatement/stockid/603843/ctrl/2016/displaytype/4.phtml
# http://money.finance.sina.com.cn/corp/go.php/vFD_CashFlow/stockid/603843/ctrl/2016/displaytype/4.phtml

# format
format.finance.summary <- function(df) {
        nms <- c("项目", "数值")
        #dt <- as.matrix(df[-c(1,2), ])
        df_out<- df
        colnames(df_out) <- nms
        return(df_out)
}


# functions to scrub summary and each statement
write.sina.finance.summary <- function(stock, statement="FinanceSummary", data_folder) {
        require(XML); require(readr)#; require(RCurl)
        if(!require(XML)) {
                install.packages("XML")
        }
        if(!require(readr)) {
                install.packages("readr")
        }
        # url in parts
        u1 <- "http://money.finance.sina.com.cn/corp/go.php/vFD_"
        u2 <- "/stockid/"
        u3 <- "/displaytype/4.phtml"
        u <- paste0(u1, statement, u2, stock, u3)
        tables <- readHTMLTable(u)
        temp <- format.finance.summary(tables[["FundHoldSharesTable"]])
        # the methods as below returns empty data
        # urldata <- getURL(u, .encoding="UTF-8")
        # data <- readHTMLTable(urldata, stringsAsFactors = FALSE)
        # View(data[["FundHoldSharesTable"]])
        
        # write file in /sina_data folder in wk folder
        output_dir <- file.path(data_folder, "sina_financials")
        if(!dir.exists(output_dir)) {
                dir.create(output_dir, showWarnings = TRUE, recursive = FALSE, mode = "0777")
        }
        write_csv(temp, path=file.path(output_dir, paste0(statement, "_", stock, "_", Sys.Date(), ".csv")))
}


#
write.sina.balance.sheet <- function(stock, yrs, statement="BalanceSheet", data_folder) {
        require(XML); require(readr)#; require(RCurl)
        if(!require(XML)) {
                install.packages("XML")
        }
        if(!require(readr)) {
                install.packages("readr")
        }
        for(yr in yrs) {
                # url in parts
                u1 <- "http://money.finance.sina.com.cn/corp/go.php/vFD_"
                u2 <- "/stockid/"
                u3 <- "/ctrl/"
                u4 <- "/displaytype/4.phtml"
                u <- paste0(u1, statement, u2, stock, u3, yr, u4)
                tables <- readHTMLTable(u)
                temp <- tables[["BalanceSheetNewTable0"]]
                # write file in /sina_data folder in wk folder
                output_dir <- file.path(data_folder, "sina_financials")
                if(!dir.exists(output_dir)) {
                        dir.create(output_dir, showWarnings = TRUE, recursive = FALSE, mode = "0777")
                }
                write_csv(as.data.frame(temp), path=file.path(output_dir, paste0(statement, "_", stock, "_", yr, ".csv")))
        }
}

#
write.sina.profit.statement <- function(stock, yrs, statement="ProfitStatement", data_folder) {
        require(XML); require(readr)#; require(RCurl)
        if(!require(XML)) {
                install.packages("XML")
        }
        if(!require(readr)) {
                install.packages("readr")
        }
        for(yr in yrs) {
                # url in parts
                u1 <- "http://money.finance.sina.com.cn/corp/go.php/vFD_"
                u2 <- "/stockid/"
                u3 <- "/ctrl/"
                u4 <- "/displaytype/4.phtml"
                u <- paste0(u1, statement, u2, stock, u3, yr, u4)
                tables <- readHTMLTable(u)
                temp <- tables[["ProfitStatementNewTable0"]]
                # write file in /sina_data folder in wk folder
                output_dir <- file.path(data_folder, "sina_financials")
                if(!dir.exists(output_dir)) {
                        dir.create(output_dir, showWarnings = TRUE, recursive = FALSE, mode = "0777")
                }
                write_csv(as.data.frame(temp), path=file.path(output_dir, paste0(statement, "_", stock, "_", yr, ".csv")))
        }
}

#
write.sina.cash.flow <- function(stock, yrs, statement="CashFlow", data_folder) {
        require(XML); require(readr)#; require(RCurl)
        if(!require(XML)) {
                install.packages("XML")
        }
        if(!require(readr)) {
                install.packages("readr")
        }
        for(yr in yrs) {
                # url in parts
                u1 <- "http://money.finance.sina.com.cn/corp/go.php/vFD_"
                u2 <- "/stockid/"
                u3 <- "/ctrl/"
                u4 <- "/displaytype/4.phtml"
                u <- paste0(u1, statement, u2, stock, u3, yr, u4)
                tables <- readHTMLTable(u)
                temp <- tables[["ProfitStatementNewTable0"]]
                # write file in /sina_data folder in wk folder
                output_dir <- file.path(data_folder, "sina_financials")
                if(!dir.exists(output_dir)) {
                        dir.create(output_dir, showWarnings = TRUE, recursive = FALSE, mode = "0777")
                }
                write_csv(as.data.frame(temp), path=file.path(output_dir, paste0(statement, "_", stock, "_", yr, ".csv")))
        }
}

# financials error handler
sina.financials.error.handler <- function(stocks, yrs, statements, data_folder) {
        tryCatch(
                {
                        message(paste("Trying", stocks, yrs, statements))
                        # can handle multiple input stocks
                        for(stock in stocks) {
                                if("FinanceSummary" %in% statements) {write.sina.finance.summary(stock, statement="FinanceSummary", data_folder)}
                                if("BalanceSheet" %in% statements) {write.sina.balance.sheet(stock, yrs, statement="BalanceSheet", data_folder)}
                                if("ProfitStatement" %in% statements) {write.sina.profit.statement(stock, yrs, statement="ProfitStatement", data_folder)}
                                if("CashFlow" %in% statements) {write.sina.cash.flow(stock, yrs, statement="CashFlow", data_folder)}
                                if(!any(statements %in% c("FinanceSummary", "BalanceSheet", "ProfitStatement", "CashFlow"))) {print("The wrong statemnts were required. Please check spelling and format of names.")}
                        }
                }, 
                # give a second try after 5s, in case of error(fail to grab data)
                error=function(stock2, yrs2, statements2) {
                        stocks2 <- stocks; yrs2 <- yrs; statements2 <- statements
                        #Sys.sleep(5)
                        #current_file <- paste(stocks2, yr2, season2)
                        #message(paste("Trying again", stocks))
                        #sina.financials.error.handler(stocks2, yrs2, statements2, data_folder)
                        message(paste("Missing", stocks, "statements"))
                        
                }
        )
}


# batch 
batch.sina.financials <- function(stocks, yrs, statements=c("FinanceSummary", "BalanceSheet", "ProfitStatement", "CashFlow"), data_folder=getwd()) {
        require(dplyr); require(XML)
        # 
        for(stock in stocks) {
                Sys.sleep(3)
                tryCatch({
                        message(paste("This is first try of", stock, paste(yrs), paste(statements)))
                        if("FinanceSummary" %in% statements) {write.sina.finance.summary(stock, statement="FinanceSummary", data_folder)}
                        if("BalanceSheet" %in% statements) {write.sina.balance.sheet(stock, yrs, statement="BalanceSheet", data_folder)}
                        if("ProfitStatement" %in% statements) {write.sina.profit.statement(stock, yrs, statement="ProfitStatement", data_folder)}
                        if("CashFlow" %in% statements) {write.sina.cash.flow(stock, yrs, statement="CashFlow", data_folder)}
                        else if(!any(statements %in% c("FinanceSummary", "BalanceSheet", "ProfitStatement", "CashFlow"))) {print("The wrong statemnts were required. Please check spelling and format of names.")}
                        
                }, 
                # give a second try after 5s, in case of error(fail to grab data)
                error=function(stock2, yrs2, statements2) {
                        stock2 <- stock; yrs2 <- yrs; statements2 <- statements
                        Sys.sleep(5)
                        message(paste("This is second try of", stock2))
                        sina.financials.error.handler(stock2, yrs2, statements2, data_folder)
                })
                
        }
}

