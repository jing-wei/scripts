library(shiny)
library(shinydashboard)
library(shinyWidgets)
library(shinyjs)
library(DT)
library(tidyverse)
library(tidyr)
library(data.table)
library(readxl)

# comment the row below to use system browser
#options(browser = file.path(dirname(dirname(getwd())), "dist/GoogleChromePortable64/GoogleChromePortable.exe") )
today <- function() format(Sys.Date(), "%Y%m%d")
today_dash <- function() format(Sys.Date(), "%Y-%m-%d")
now <- function() format(Sys.time(), "%H:%M")
now2 <- function() format(Sys.time(), "%H%M")

# DT functions

dt_output = function(title, id) {
  fluidRow(column(
    #12, h4(paste0('Table ', sub('.*?([0-9]+)$', '\\1', id), ': ', title)),
    12, h4(paste0('Table', ': ', title)),
    hr(), DTOutput(id)#, width = 900)
  ))
}
render_dt = function(data, editable = 'cell', server = TRUE, ...) {
  renderDT(data, selection = 'none', server = server, editable = editable, ...)
}


##################
