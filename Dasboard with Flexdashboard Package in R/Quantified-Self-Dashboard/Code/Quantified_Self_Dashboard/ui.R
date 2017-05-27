#
# This is the user-interface definition of a Shiny web application. You can
# run the application by clicking 'Run App' above.
#
# Find out more about building applications with Shiny here:
# 
#    http://shiny.rstudio.com/
#

library(shiny)
library(shinythemes)
require(DT)

# Define UI for application that draws a histogram
shinyUI(navbarPage(
  
  # Application title
  title = "Quantified Self Dashboard",

  tabPanel("Days", DT::dataTableOutput('dt_days')),
  tabPanel("Days Plot", plotOutput("plot_days")),
  tabPanel("Pomodoros", DT::dataTableOutput('dt_pomodoros'))
))
