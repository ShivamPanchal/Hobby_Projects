  #
# This is the server logic of a Shiny web application. You can run the 
# application by clicking 'Run App' above.
#
# Find out more about building applications with Shiny here:
# 
#    http://shiny.rstudio.com/
#

library(shiny)
require(DT)
require(tidyverse)
require(ggthemes)
require(rprojroot)

setwd(rprojroot::find_root("Quantified Self Dashboard.Rproj"))
days = read_csv("Data/Input/days_16.csv")
pomodoros = read_csv("Data/Input/pomodoros_16.csv")

source("Code/Utility Functions.R")

# Define server logic required to draw a histogram
shinyServer(function(input, output) {
   
  output$plot_days <- renderPlot({
    
    plotDays()
    
  })
  
  output$dt_days = DT::renderDataTable({
    DT::datatable(days)
  })
  
  output$dt_pomodoros = DT::renderDataTable({
    DT::datatable(pomodoros)
  })
  
})
