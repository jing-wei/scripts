


####

server = function(input, output, session) {
  
  session$onSessionEnded(function() {
    stopApp()
  })
  
  
  # Source functions ---------------------------------------------------------
  
  
  
  # importReception ---------------------------------------------------------
  # 
  load_reception <- function(input) {
    inFile <- input$reception$datapath
    inFileName = input$reception$name
    #
    out <- read_csv(inFile) # adjust for your input file
    return(out)
    
  }
  
  
  observeEvent(input$uploadReception, {
    reception <- load_reception(input)
    
    output$submitReceptionTbl <- renderDT(
      reception, selection = 'none', editable = TRUE, 
      rownames = TRUE, 
      extensions = 'FixedColumns',
      
      options = list(
        paging = TRUE,
        searching = TRUE,
        #lengthMenu = list(c(50, 100, -1), c('50', '100', 'All')),
        pageLength = 20, 
        autoWidth = TRUE,
        ordering = TRUE,
        dom = 'Bfrtip',
        fixedColumns = list(leftColumns = 2),
        scrollX = TRUE
      ),
      
      #filter = "top", 
      class = "display"
    )
    
    
    observeEvent(input$submitReceptionTbl_cell_edit, {
      reception[input$submitReceptionTbl_cell_edit$row,input$submitReceptionTbl_cell_edit$col] <<- input$submitReceptionTbl_cell_edit$value
      #reception <<- editData(reception, input$submitReceptionTbl_cell_edit, 'submitReceptionTbl')
    })
    
    #
    view_fun<-eventReactive(input$uploadReception,{
      if(is.null(input$submitReception)||input$submitReception==0)
      {
        returnValue()
      }
      else
      {
        DT::datatable(reception,selection = 'none')
      }
      
    })
    output$receptionSubmitMessage <- NULL
    save_dir <- input$Output_Location
    observeEvent(input$submitReception,{
      outReceptionName <- str_replace(unique(reception$`#Reception`), "\\.xlsx", "\\.csv")
      fwrite(reception,file = file.path(save_dir, "reception_submitted", outReceptionName))
      output$receptionSubmitMessage <- renderText({ paste0("Reception File Submitted and Saved as: ", outReceptionName) })
    })
    # 
    output$updated.df_Reception<-renderDataTable({
      view_fun()
    }
    )
  })
  
  
  
  
  
  
  # importRun ---------------------------------------------------------
  # 
  load_SB <- function(input) {
    inFile <- input$transferSummary$datapath
    inFileName = input$transferSummary$name
    #
    out <- read_csv(inFile) # adjust for your input file
    return(out)
  }
  load_interpretive <- function(input) {
    inFile <- input$interpretive$datapath
    inFileName = input$interpretive$name
    #
    out <- read_csv(inFile) # adjust for your input file
    return(out)
  }
  
  
  observeEvent(input$uploadRun, {
    tSummary <- load_SB(input)
    interpretive <- load_interpretive(input)
    runCombined <- tSummary %>% 
      left_join(interpretive , 
                by=c("Run ID"="#Plate", "PCR Well ID"="Well"))
    
    output$submitRunTbl <- renderDT(
      runCombined, selection = 'none', editable = TRUE, 
      rownames = TRUE, 
      extensions = 'FixedColumns',
      
      options = list(
        paging = TRUE,
        searching = TRUE,
        #lengthMenu = list(c(50, 100, -1), c('50', '100', 'All')),
        pageLength = 20, 
        autoWidth = TRUE,
        ordering = TRUE,
        dom = 'Bfrtip',
        fixedColumns = list(leftColumns = 2),
        scrollX = TRUE
      ),
      
      #filter = "top", 
      class = "display"
    )
    
    
    observeEvent(input$submitRunTbl_cell_edit, {
      runCombined[input$submitRunTbl_cell_edit$row,input$submitRunTbl_cell_edit$col] <<- input$submitRunTbl_cell_edit$value
      
    })
    
    #
    view_fun<-eventReactive(input$uploadRun,{
      if(is.null(input$submitRun)||input$submitRun==0)
      {
        returnValue()
      }
      else
      {
        DT::datatable(runCombined,selection = 'none')
      }
      
    })
    output$runSubmitMessage <- NULL
    save_dir <- input$Output_Location_Run
    observeEvent(input$submitRun,{
      outRunName <- str_replace(unique(runCombined$`#SampleBatching`), "\\.xlsx", "\\.csv")
      fwrite(runCombined,file = file.path(save_dir, "run_submitted", outRunName))
      output$runSubmitMessage <- renderText({ paste0("Run Data Submitted and Saved as: ", outRunName) })
      
    })
    # 
    output$updated.df_Run<-renderDataTable({
      view_fun()
    }
    )
  })
  
  # Import Disposal ---------------------------------------------------------
  # 
  load_disposal <- function(input) {
    inFile <- input$disposal$datapath
    inFileName = input$disposal$name
    #
    out <- read_csv(inFile) # adjust for your input file
    return(out)
    
  }
  
  
  observeEvent(input$uploadDisposal, {
    disposal <- load_disposal(input)
    
    output$submitDisposalTbl <- renderDT(
      disposal, selection = 'none', editable = TRUE, 
      rownames = TRUE, 
      extensions = 'FixedColumns',
      
      options = list(
        paging = TRUE,
        searching = TRUE,
        autoWidth = FALSE, 
        #lengthMenu = list(c(50, 100, -1), c('50', '100', 'All')),
        pageLength = 20, 
        autoWidth = TRUE,
        ordering = TRUE,
        dom = 'Bfrtip',
        fixedColumns = FALSE,
        scrollX = TRUE
      ),
      
      #filter = "top", 
      class = "display"
    )
    
    
    observeEvent(input$submitDisposalTbl_cell_edit, {
      disposal[input$submitDisposalTbl_cell_edit$row,input$submitDisposalTbl_cell_edit$col] <<- input$submitDisposalTbl_cell_edit$value
      
    })
    
    #
    view_fun<-eventReactive(input$uploadDisposal,{
      if(is.null(input$submitDisposal)||input$submitDisposal==0)
      {
        returnValue()
      }
      else
      {
        DT::datatable(disposal,selection = 'none')
      }
      
    })
    output$disposalSubmitMessage <- NULL
    save_dir <- input$Output_Location_Disposal
    observeEvent(input$submitDisposal,{
      outDisposalName <- str_replace(unique(disposal$`#Disposal`), "\\.xlsx", "\\.csv")
      fwrite(disposal,file = file.path(save_dir, "disposal_submitted", outDisposalName))
      output$disposalSubmitMessage <- renderText({ paste0("Disposal Data Submitted and Saved as: ", outDisposalName) })
    })
    # 
    output$updated.df_Disposal<-renderDataTable({
      view_fun()
    }
    )
  })
  
  
  # Import Return Shipment ---------------------------------------------------------
  load_return <- function(input) {
    inFile <- input$return$datapath
    inFileName = input$return$name
    #
    out <- read_csv(inFile) # adjust for your input file
    return(out)
    
  }
  
  
  observeEvent(input$uploadReturn, {
    return <- load_return(input)
    
    output$submitReturnTbl <- renderDT(
      return, selection = 'none', editable = TRUE, 
      rownames = TRUE, 
      extensions = 'FixedColumns',
      
      options = list(
        paging = TRUE,
        searching = TRUE,
        autoWidth = FALSE, 
        #lengthMenu = list(c(50, 100, -1), c('50', '100', 'All')),
        pageLength = 20, 
        autoWidth = TRUE,
        ordering = TRUE,
        dom = 'Bfrtip',
        fixedColumns = FALSE,
        scrollX = TRUE
      ),
      
      #filter = "top", 
      class = "display"
    )
    
    
    observeEvent(input$submitReturnTbl_cell_edit, {
      return[input$submitReturnTbl_cell_edit$row,input$submitReturnTbl_cell_edit$col] <<- input$submitReturnTbl_cell_edit$value
      
    })
    
    #
    view_fun<-eventReactive(input$uploadReturn,{
      if(is.null(input$submitReturn)||input$submitReturn==0)
      {
        returnValue()
      }
      else
      {
        DT::datatable(return,selection = 'none')
      }
      
    })
    output$returnSubmitMessage <- NULL
    save_dir <- input$Output_Location_Return
    observeEvent(input$submitReturn,{
      outReturnName <- str_replace(unique(return$`#Return`), "\\.xlsx", "\\.csv")
      fwrite(return,file = file.path(save_dir, "return_submitted", outReturnName))
      output$returnSubmitMessage <- renderText({ paste0("Return Shipment Data Submitted and Saved as: ", outReturnName) })
    })
    # 
    output$updated.df_Return<-renderDataTable({
      view_fun()
    }
    )
  })
  
  # Sample Tracker ---------------------------------------------------------
  observeEvent(input$retrieveData, {
    data_dir <- input$Data_Location
    
    files_reception <- file.info(list.files(file.path(data_dir, "reception_submitted"))) %>% 
      tibble::rownames_to_column("File") %>% 
      dplyr::select(c("File", "ctime"))
    output$files_reception <- renderDT(
      files_reception, 
      selection = 'none', editable = FALSE, 
      rownames = TRUE, 
      extensions = 'FixedColumns',
      
      options = list(
        paging = TRUE,
        searching = TRUE,
        autoWidth = FALSE, 
        #lengthMenu = list(c(50, 100, -1), c('50', '100', 'All')),
        pageLength = 10, 
        autoWidth = TRUE,
        ordering = TRUE,
        dom = 'Bfrtip',
        fixedColumns = FALSE,
        scrollX = TRUE
      ),
      
      #filter = "top", 
      class = "display"
    )
    
    files_run <- file.info(list.files(file.path(data_dir, "run_submitted"))) %>% 
      tibble::rownames_to_column("File") %>% 
      dplyr::select(c("File", "ctime"))
    output$files_run <- renderDT(
      files_run, 
      selection = 'none', editable = FALSE, 
      rownames = TRUE, 
      extensions = 'FixedColumns',
      
      options = list(
        paging = TRUE,
        searching = TRUE,
        autoWidth = FALSE, 
        #lengthMenu = list(c(50, 100, -1), c('50', '100', 'All')),
        pageLength = 10, 
        autoWidth = TRUE,
        ordering = TRUE,
        dom = 'Bfrtip',
        fixedColumns = FALSE,
        scrollX = TRUE
      ),
      
      #filter = "top", 
      class = "display"
    )  
    
    files_disposal <- file.info(list.files(file.path(data_dir, "disposal_submitted"))) %>% 
      tibble::rownames_to_column("File") %>% 
      dplyr::select(c("File", "ctime"))
    output$files_disposal <- renderDT(
      files_disposal, 
      selection = 'none', editable = FALSE, 
      rownames = TRUE, 
      extensions = 'FixedColumns',
      
      options = list(
        paging = TRUE,
        searching = TRUE,
        autoWidth = FALSE, 
        #lengthMenu = list(c(50, 100, -1), c('50', '100', 'All')),
        pageLength = 10, 
        autoWidth = TRUE,
        ordering = TRUE,
        dom = 'Bfrtip',
        fixedColumns = FALSE,
        scrollX = TRUE
      ),
      
      #filter = "top", 
      class = "display"
    )
    
    files_return <- file.info(list.files(file.path(data_dir, "return_submitted"))) %>% 
      tibble::rownames_to_column("File") %>% 
      dplyr::select(c("File", "ctime"))
    output$files_return <- renderDT(
      files_return, 
      selection = 'none', editable = TRUE, 
      rownames = TRUE, 
      extensions = 'FixedColumns',
      
      options = list(
        paging = TRUE,
        searching = TRUE,
        autoWidth = FALSE, 
        #lengthMenu = list(c(50, 100, -1), c('50', '100', 'All')),
        pageLength = 10, 
        autoWidth = TRUE,
        ordering = TRUE,
        dom = 'Bfrtip',
        fixedColumns = FALSE,
        scrollX = TRUE
      ),
      
      #filter = "top", 
      class = "display"
    )
      
  })
  
  # generate summary
  
  ## to do. 
  ## Add filtering of data and export
  ## could add restrictions like date range (only past ten days?) to avoid too large a file
  
  observeEvent(input$generateSummary, {
    data_dir <- input$Data_Location
    receptionData <- list.files(file.path(data_dir, "reception_submitted"), full.names = TRUE) %>% 
      map( ~ fread(file = .x, sep = ",", colClasses = "text")) %>% 
      bind_rows()
    
    runData <- list.files(file.path(data_dir, "run_submitted"), full.names = TRUE) %>% 
      map( ~ fread(file = .x, sep = ",", colClasses = "text")) %>% 
      bind_rows()
    disposalData <- list.files(file.path(data_dir, "disposal_submitted"), full.names = TRUE) %>% 
      map( ~ fread(file = .x, sep = ",", colClasses = "text")) %>% 
      bind_rows()
    returnData <- list.files(file.path(data_dir, "return_submitted"), full.names = TRUE) %>% 
      map( ~ fread(file = .x, sep = ",", colClasses = "text")) %>% 
      bind_rows()
    
    sampleSummary <- receptionData  %>% 
      full_join(runData, 
                by=c("Sample ID"="Sample ID",
                     "Site ID"="Site ID")) %>% 
      left_join(disposalData, 
                by=c("Sample ID"="Disposal Sample ID")) %>% 
      left_join(returnData, 
                by = c("Sample ID"="Returned Sample ID"))
    
    
    output$sampleSummaryTbl <- renderDT(
      sampleSummary, selection = 'none', editable = TRUE, 
      rownames = FALSE, 
      extensions = 'FixedColumns',
      server = TRUE, # this table could be huge
      options = list(
        paging = TRUE,
        searching = TRUE,
        lengthMenu = list(c(50, 100, -1), c('50', '100', 'All')),
        pageLength = 20, 
        autoWidth = TRUE,
        ordering = TRUE,
        dom = 'Bfrtip',
        fixedColumns = list(leftColumns = 2),
        scrollX = TRUE
      ),
      
      filter = "top", 
      class = "display"
    )
    # save
    observeEvent(input$saveSummary, {
      # Save a copy for dash board, not dated
      out_summary_file <- paste0("Sample_Summary", ".csv")
      fwrite(sampleSummary, file = file.path(data_dir, "summary", out_summary_file))
      out_summary_file <- paste0("Sample_Summary_", today(), now(), ".csv")
      fwrite(sampleSummary, file = file.path(data_dir, "summary", out_summary_file))
    })
  })
  
  # Dash Board ---------------------------------------------------------
  #observeEvent(input$generateDashboard, {
  #  summary_dir <- input$Data_Location_Dashboard
  #  dashboardData <- fread(file = file.path(summary_dir, "summary", "Sample_Summary.csv"))
  #  output$sampleReceivedDaily <- dashboardData # to add analysis
  #})
  
  ####
  
  
}




