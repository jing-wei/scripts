library(dplyr)
library(hflights)

# load data
data(hflights)
str(hflights)
head(hflights)

# convert to local data frame
flights <- tbl_df(hflights)

head(flights, 10)

# back to normal data frame
data.frame(head(flights))
### filter
# to view all flights on january 1
flights[flights$Month == 1 & flights$DayofMonth ==1,]

# dplyr approach
filter(flights, Month == 1, DayofMonth == 1)

# pipe for OR condition
filter(flights, UniqueCarrier=="AA" | UniqueCarrier=="UA")
# %in%
filter(flights, UniqueCarrier %in% c("AA", "UA"))
### select
# base R approach
flights[, c("DepTime", "ArrTime", "FlightNum")]
# dplyr
select(flights, DepTime, ArrTime, FlightNum)
# for multiple contiguous columns, use colon. Use "contains" to match
# column by name
# others: starts_with, ends_with, matches (for regular expression)
select(flights, Year:DayofMonth, contains("Taxi"), contains("Delay"))

### chaining or piping
# nesting
# to select uniquecarrier and depdelay columns and filter for delays over 60minuts
filter(select(flights, UniqueCarrier, DepDelay), DepDelay > 60)
# chaining
flights %>% select(UniqueCarrier, DepDelay) %>% 
        filter(DepDelay > 60)

### arrange: reorder rows
# base R approach
flights[order(flights$DepDelay), c("UniqueCarrier", "DepDelay")]
# dplyr
flights %>% select(UniqueCarrier, DepDelay) %>% 
        arrange(desc(DepDelay))

### mutate: add new variables
# base R approach
flights$Speed <- flights$Distance / flights$AirTime * 60
flights[, c("Distance", "AirTime", "Speed")]
# dplyr
flights %>% select(Distance, AirTime) %>% 
        mutate(Speed = Distance/AirTime*60)
# store the new variable
flights <- flights %>% mutate(Speed=Disance/AirTime*60)

### summarise: reduce variables to values
# primarily useful with data that has been grouped by one or more
# variables. 
# "group_by" creates groups that will be operated on
# "summarise" uses the provided aggregation function to summarise each group
# base R approach
head(with(flights, tapply(ArrDelay, Dest, mean, na.rm=T)))
head(aggregate(ArrDelay ~ Dest, flights, mean))
# dplyr
flights %>% 
        group_by(Dest) %>% 
        summarise(avg_delay = mean(ArrDelay, na.rm=T))
# summarise_each: apply the same summary function to multi columns
# mutate_each
flights %>% group_by(UniqueCarrier) %>% 
        summarise_each(funs(mean), Cancelled, Diverted)
# 
flights %>% group_by(UniqueCarrier) %>% 
        summarise_each(funs(min(., na.rm=T), max(., na.rm=T)), 
                       matches("Delay"))

### helper functions
flights %>% group_by(Month, DayofMonth) %>% 
        summarise(flight_count=n()) %>% 
        arrange(desc(flight_count))
# 
flights %>% group_by(Month, DayofMonth) %>% 
        tally(sort=T)
# 
flights %>% group_by(Dest) %>% 
        summarise(flight_count=n(), plane_count=n_distinct(TailNum))
# grouping without summarising
flights %>% group_by(Dest) %>% 
        select(Cancelled) %>% 
        table() %>% head()

### window functions
# window function, takes n input and returns n values
# ranking functions: min_rank, 
# offset functions: lead, lag, 
# cumulative aggregates: cummean, 
flights %>% group_by(UniqueCarrier) %>% 
        select(Month, DayofMonth, DepDelay) %>% 
        filter(min_rank(desc(DepDelay)) <= 2) %>% 
        arrange(UniqueCarrier, desc(DepDelay))
# simplify with top_n()
flights %>% group_by(UniqueCarrier) %>% 
        select(Month, DayofMonth, DepDelay) %>% 
        top_n(2) %>% 
        arrange(UniqueCarrier, desc(DepDelay))
# 
flights %>% group_by(Month) %>% 
        summarise(flight_count = n()) %>% 
        mutate(change = flight_count - lag(flight_count))
# 
flights %>% group_by(Month) %>% 
        tally() %>% 
        mutate(change = n - lag(n))

### other useful functions
# randomly sample without replacement
flights %>% sample_n(5)
# randomly sample a fractin of rows
flights %>% sample_frac(0.2, replace=T)
# dplyr approach to str()
glimpse(flights)

### connecting to database
# Currently support SQLite, PostgreSQL/Redshift, MySQL/MariaDB
# Instructions to create database are in dplyr vignettes
my_db <- src_sqlite("my_db.sqlite3")
