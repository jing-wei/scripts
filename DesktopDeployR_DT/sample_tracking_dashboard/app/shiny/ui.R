


dashboardPage(

  dashboardHeader(title = "App"),
  
  # Sidebar -----------------------------------------------------------------
  
  dashboardSidebar(
    sidebarMenu(
      id = "tabs",
      menuItem(text = "Import Reception", tabName = "importReception", icon = icon("file-upload")),
      menuItem(text = "Import Run", tabName = "importRun", icon = icon("pencil")),
      menuItem(text = "Import Disposal", tabName = "importDisposal", icon = icon("trash-alt")),
      menuItem(text = "Import Return Shipment", tabName = "importReturn", icon = icon("reply-all")),
      menuItem("Sample Tracker", tabName = "sampleTracker", icon = icon("chart-line")),
      
      menuItem("Dashboard", tabName = "dashboard", icon = icon("tachometer-alt")),
      menuItem("About", tabName = "about", icon = icon("info-circle"))#,
      
    )
  ),
  
  dashboardBody(
    tags$style(".bttn-gradient.bttn-default  {color: #5D8CB9 !important;}"),
    tags$style(HTML("hr {border-top: 1px solid #000000;}")),
    useSweetAlert(),
    useShinyjs(),  # Include shinyjs
    tabItems(
      # Import -----------------------------------------------------------------
      tabItem(tabName = "importReception",
              
              fileInput(
                inputId = "reception", 
                width = "400px", 
                label = "Import reception files with incoming sample information", 
                multiple = FALSE, 
                buttonLabel = "Browse...", 
                placeholder = "No file selected", 
                accept = c(".xlsx")), 
              
              actionButton("uploadReception", "Upload Reception File"), 
              
              fluidPage(
                dt_output('Please verify data before submitting (Double click to edit)', 'submitReceptionTbl')
              ), 
              
              textInput(inputId = "Output_Location", 
                        width = "400px", 
                        label = "Please do not modify folder path below unless necessary", 
                        value = "C:\\Users\\data"), 
              br(), verbatimTextOutput("receptionSubmitMessage"), 
              actionButton("submitReception", "Submit Reception File"), 
              DTOutput("updated.df_Reception")
              
      ), 
      
      # Import Run -----------------------------------------------------------------
      tabItem(tabName = "importRun",
              
              fileInput(
                inputId = "transferSummary", 
                width = "400px", 
                label = "Import sample batching file for a run", 
                multiple = FALSE, 
                buttonLabel = "Browse...", 
                placeholder = "No file selected", 
                accept = c(".xlsx")), 
              fileInput(
                inputId = "interpretive", 
                width = "400px", 
                label = "Import results from interpretive software for the SAME run", 
                multiple = FALSE, 
                buttonLabel = "Browse...", 
                placeholder = "No file selected", 
                accept = c(".csv")), 
              
              actionButton("uploadRun", "Upload Run Data"), 
              # 
              fluidPage(
                dt_output('Please verify data before submitting (Double click to edit)', 'submitRunTbl')
              ), 
              
              textInput(inputId = "Output_Location_Run", 
                        width = "400px", 
                        label = "Please do not modify folder path below unless necessary", 
                        value = "C:\\Users\\data"), 
              br(), verbatimTextOutput("runSubmitMessage"), 
              
              actionButton("submitRun", "Submit Run Data"), 
              DTOutput("updated.df_Run")
      ), 
      
      # Import Disposal -----------------------------------------------------------------
      tabItem(tabName = "importDisposal",
              
              fileInput(
                inputId = "disposal", 
                width = "400px", 
                label = "Import information of sample disposal", 
                multiple = FALSE, 
                buttonLabel = "Browse...", 
                placeholder = "No file selected", 
                accept = c(".xlsx")),
              
              actionButton("uploadDisposal", "Upload Disposal Data"), 
              # 
              fluidPage(
                dt_output('Please verify data before submitting (Double click to edit)', 'submitDisposalTbl')
              ), 
              
              textInput(inputId = "Output_Location_Disposal", 
                        width = "400px", 
                        label = "Please do not modify folder path below unless necessary", 
                        value = "C:\\Users\\data"), 
              
              br(), verbatimTextOutput("disposalSubmitMessage"), 
              
              actionButton("submitDisposal", "Submit Disposal Data"), 
              DTOutput("updated.df_Disposal")
      ), 
      
      # Import Return Shipment -----------------------------------------------------------------
      tabItem(tabName = "importReturn",
              
              fileInput(
                inputId = "return", 
                width = "400px", 
                label = "Import information of sample return", 
                multiple = FALSE, 
                buttonLabel = "Browse...", 
                placeholder = "No file selected", 
                accept = c(".xlsx")),
              
              actionButton("uploadReturn", "Upload Return Shipment Data"), 
              # 
              fluidPage(
                dt_output('Please verify data before submitting (Double click to edit)', 'submitReturnTbl')
              ), 
              
              textInput(inputId = "Output_Location_Return", 
                        width = "400px", 
                        label = "Please do not modify folder path below unless necessary", 
                        value = "C:\\Users\\data"), # adjust to folder containing example data
              
              br(), verbatimTextOutput("returnSubmitMessage"), 
              
              actionButton("submitReturn", "Submit Return Shipment Data"), 
              DTOutput("updated.df_Return")
      ), 
      
      
      
      #### Sample Tracker -----------------------------------------------------------------
      tabItem(tabName = "sampleTracker",
              h4("Summary of testing samples will be based on correct and up-to-date information submitted as below!"), 
              h4("Please verify before proceeding."), 
              br(), 
              textInput(inputId = "Data_Location", 
                        width = "400px", 
                        label = "Please do not modify folder path below unless necessary", 
                        value = "C:\\Users\\data"), 
              
              br(), 
              fluidRow(
                column(width = 3, 
                       actionButton("retrieveData", "Retrieve Submitted Data & List Below")), 
                column(width = 3, 
                       actionButton("generateSummary", "Generate Sample Summary")), 
                column(width = 4, 
                       actionButton("saveSummary", "Export copy of Sample Summary up to now"))
              ), 
              
              fluidRow(
                column(width = 6, 
                       dt_output('Submitted Reception Information:', 'files_reception')), 
                column(width = 6, 
                       dt_output('Submitted Run Data:', 'files_run'))
                ), 
              fluidRow(
                column(width = 6, 
                       dt_output('Submitted Disposal Information:', 'files_disposal')), 
                column(width = 6, 
                       dt_output('Submitted Return Shipment Information:', 'files_return')
                )
              ), 
              
              fluidPage(
                dt_output("Sample Summary: ", "sampleSummaryTbl")
              )
      ), 
      
      #### Dash Board ---------------------------------------------------------------------
      ## plot in two tabs: by day or week?
      #tabItem(tabName = "dashboard",
      #  h4("Summary of testing samples will be based on correct and up-to-date information submitted as below!"), 
      #  h4("Please verify before proceeding."), 
      #  br(), 
      #  textInput(inputId = "Data_Location_Dashboard", 
      #            width = "400px", 
      #            label = "Please do not modify folder path below unless necessary", 
      #            value = "C:\\Users\\data"), 
      #  
      #  br(),
      #  actionButton("generateDashboard", "Generate Dashboard"), 
      #  tabsetPanel(
      #    tabPanel("dailyTable", 
      #             # Samples received. AM/PM and total
      #             tableOutput("sampleReceivedDaily")
      #             ), 
      #    tabPanel("weeklyTable", 
      #             # Samples received. AM/PM and total
      #             tableOutput("sampleReceivedDaily") # Update to weekly later
      #             )
      #  )
      #), 
      
      
      
      # Samples Analyzed
      
      # Data Delivered
      
      # Samples Left to Analyze
      
      # Repeat Samples. Not reported. 1st inconclusive
      
      # Repeat Samples. Reported. 2nd inconclusive
      
      # Repeat Samples. Not reported. Invalid
      
      # Samples of Discrepancy. Received (not processed?)
      
      #### About --------------------------------------------------------------------------
      
      tabItem(tabName = "about",
              includeCSS("custom-css.css"),
              includeMarkdown("about.md"))
      
    )
  )
)

