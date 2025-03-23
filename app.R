# -----------------------------------------------------------------------------
# Project: Electricity consumption analysis and forecasting in France
# Author : Felipe Alvarez Suarez
# Date : 15/11/2024
# -----------------------------------------------------------------------------
# Description:
# This project uses R and Shiny to analyze and forecast electricity consumption
# in France from 2012 to 2024. The application offers:
#   - An exploratory analysis (interactive charts for visualization)
#   - Forecast models based on SARIMA and SNaive
#   - An interactive system to adjust SARIMA parameters
#
# Organization:
# 1) Load libraries
# 2) Data preprocessing
# 3) Auxiliary functions
# 4) Shiny UI
# 5) Shiny server
# 6) Run the application
# -----------------------------------------------------------------------------

# ------------------------------------------------------------------------- 
# 1) Load libraries
# ------------------------------------------------------------------------- 

conflictRules("dplyr", exclude = "lag")

suppressPackageStartupMessages({
  library(shiny)  
  library(shinythemes)  
  library(dplyr)  
  library(lubridate)  
  library(ggplot2)  
  library(plotly)  
  library(forecast)  
  library(tseries)  
  library(ggfortify)  
  library(scales)  
  library(xts)  
  library(Metrics)  
  library(MuMIn)  
})

# Force time locale to English for date labels:
Sys.setlocale("LC_TIME", "en_US.UTF-8")

# ------------------------------------------------------------------------- 
# 2) Data loading / preprocessing
# ------------------------------------------------------------------------- 

if (file.exists("data_nettoyees.Rdata")) {  
  load("data_nettoyees.Rdata")  
} else {  
  data <- read.csv("consommation-quotidienne-brute.csv", sep = ";")  
  str(data)
  
  data$Date_Heure <- as.POSIXct(
    paste(data$Date, data$Heure),
    format = "%d/%m/%Y %H:%M",
    tz = "Europe/Paris"
  )
  
  data$Date <- as.Date(
    data$Date,
    format = "%d/%m/%Y"
  )
  data$Heure <- format(
    strptime(data$Heure, format = "%H:%M"),
    "%H:%M"
  )
  
  # Rename the consumption column and keep only necessary columns
  data <- data %>%
    select(
      Date_Heure,
      Date,
      Heure,
      Conso_brute_electricite_MW = Consommation.brute.électricité..MW....RTE
    )
  
  total_lignes_avant <- nrow(data)
  
  # Remove rows with missing date/time or consumption
  data <- data[
    !is.na(data$Date_Heure) & !is.na(data$Conso_brute_electricite_MW),  
  ]
  
  total_lignes_apres <- nrow(data)
  lignes_supprimees <- total_lignes_avant - total_lignes_apres
  message(sprintf("Number of removed values: %d out of %d rows.",
                  lignes_supprimees, total_lignes_avant))
  
  # Daily aggregation
  data_daily <- aggregate(
    Conso_brute_electricite_MW ~ Date,
    data,
    FUN = sum
  )
  names(data_daily)[2] <- "Conso_totale_journaliere_MW"
  
  data_daily$Mois <- format(
    as.Date(data_daily$Date, format = "%d/%m/%Y"),
    "%Y-%m"
  )
  
  # Monthly aggregation
  data_monthly <- aggregate(
    Conso_totale_journaliere_MW ~ Mois,
    data_daily,
    FUN = sum
  )
  names(data_monthly)[2] <- "Conso_totale_mensuelle_MW"
  
  # For monthly data, add month number and average daily consumption
  data_monthly$mois_num <- month(
    as.Date(paste0(data_monthly$Mois, "-01")),
    label = TRUE, abbr = TRUE, locale = "fr_FR"  # original loading 
  )
  data_monthly$days_in_month <- days_in_month(
    as.Date(paste0(data_monthly$Mois, "-01"))
  )
  data_monthly$Conso_moy_jour_MW <- (
    data_monthly$Conso_totale_mensuelle_MW / data_monthly$days_in_month
  )
  
  # Monthly time series starting 2012, frequency=12
  mois_ts <- ts(
    data_monthly$Conso_totale_mensuelle_MW,
    frequency = 12,
    start = c(2012, 1)
  )
  
  # Train data up to April 2022
  train_data <- window(mois_ts, end = c(2022, 4))
  # Validation data: May 2022 - April 2023
  refine_data <- window(mois_ts, start = c(2022, 5), end = c(2023, 4))
  # Combined train+validation: up to April 2023
  combined_train_refine <- window(mois_ts, end = c(2023, 4))
  # Test data: May 2023 - April 2024
  test_data <- window(mois_ts, start = c(2023, 5), end = c(2024, 4))
  
  save(
    data,
    data_daily,
    data_monthly,
    mois_ts,
    train_data,
    refine_data,
    combined_train_refine,
    test_data,
    file = "data_nettoyees.Rdata"
  )
}

# ------------------------------------------------------------------------- 
# 3) Auxiliary functions
# ------------------------------------------------------------------------- 

# 3.a) Histogram of residuals 
plot_residuals_histogram <- function(residuals) {  
  residuals <- as.numeric(residuals)
  N <- length(residuals)
  
  df_residus <- data.frame(
    index = 1:N,
    residual = residuals
  )
  
  mean_res <- mean(df_residus$residual)
  sd_res <- sd(df_residus$residual)
  hist_data <- hist(
    df_residus$residual,
    plot = FALSE,
    breaks = 30
  )
  
  plot_ly() %>%
    add_bars(
      x = hist_data$mids,
      y = hist_data$density,
      name = "Histogram",
      opacity = 0.6
    ) %>%
    add_lines(
      x = seq(
        min(df_residus$residual),
        max(df_residus$residual),
        length.out = 100
      ),
      y = dnorm(
        seq(min(df_residus$residual), max(df_residus$residual), length.out = 100),
        mean = mean_res,
        sd = sd_res
      ),
      name = "Normal curve",
      line = list(color = 'red')
    ) %>%
    layout(
      title = "Histogram of residuals with normal curve",
      xaxis = list(title = "Residuals"),
      yaxis = list(title = "Density"),
      hovermode = 'x unified'
    )
}

# 3.b) QQ plot of residuals
plot_qq_residuals <- function(residuals) {
  residuals <- as.numeric(residuals)
  N <- length(residuals)
  
  df_residus <- data.frame(
    index = 1:N,
    residual = residuals
  )
  
  ggplotly(
    ggplot(
      df_residus,
      aes(sample = residual)
    ) +
      stat_qq(shape = 1, color = "purple") +
      stat_qq_line(color = "red") +
      theme_minimal() +
      labs(
        title = "QQ plot of residuals",
        x = "Theoretical quantiles",
        y = "Sample quantiles"
      ) +
      theme(plot.title = element_text(hjust = 0.5))
  )
}

# 3.c) Residual analysis function
analyse_residus <- function(model, lag_max = 50, title_prefix = "SARIMA residuals") {
  residus <- residuals(model)
  residus <- as.numeric(residus)
  N <- length(residus)
  
  df_residus <- data.frame(
    index = 1:N,
    residual = residus
  )
  
  sw_test <- shapiro.test(df_residus$residual)
  lb_test <- Box.test(
    df_residus$residual,
    lag = lag_max,
    type = "Ljung-Box"
  )
  
  acf_data <- acf(df_residus$residual, lag.max = lag_max, plot = FALSE)
  df_acf <- data.frame(
    lag = acf_data$lag[-1],
    acf = acf_data$acf[-1]
  )
  acf_conf <- 1.96 / sqrt(N)
  
  pacf_data <- pacf(df_residus$residual, lag.max = lag_max, plot = FALSE)
  df_pacf <- data.frame(
    lag = pacf_data$lag,
    pacf = pacf_data$acf
  )
  pacf_conf <- 1.96 / sqrt(N)
  
  qq_plotly <- plot_qq_residuals(df_residus$residual)
  hist_plotly <- plot_residuals_histogram(df_residus$residual)
  
  p_acf <- plot_ly() %>%
    add_bars(
      x = df_acf$lag,
      y = df_acf$acf,
      name = "ACF",
      marker = list(color = "blue")
    ) %>%
    add_lines(
      x = df_acf$lag,
      y = rep(acf_conf, length(df_acf$lag)),
      name = "IC +",
      line = list(color = 'red', dash = 'dash')
    ) %>%
    add_lines(
      x = df_acf$lag,
      y = rep(-acf_conf, length(df_acf$lag)),
      name = "IC -",
      line = list(color = 'red', dash = 'dash')
    ) %>%
    layout(
      title = paste(title_prefix, "- ACF"),
      xaxis = list(title = "Lag"),
      yaxis = list(title = "ACF")
    )
  
  p_pacf <- plot_ly() %>%
    add_bars(
      x = df_pacf$lag,
      y = df_pacf$pacf,
      name = "PACF",
      marker = list(color = "green")
    ) %>%
    add_lines(
      x = df_pacf$lag,
      y = rep(pacf_conf, length(df_pacf$lag)),
      name = "IC +",
      line = list(color = 'red', dash = 'dash')
    ) %>%
    add_lines(
      x = df_pacf$lag,
      y = rep(-pacf_conf, length(df_pacf$lag)),
      name = "IC -",
      line = list(color = 'red', dash = 'dash')
    ) %>%
    layout(
      title = paste(title_prefix, "- PACF"),
      xaxis = list(title = "Lag"),
      yaxis = list(title = "PACF")
    )
  
  return(
    list(
      acf_plot = p_acf,
      pacf_plot = p_pacf,
      qq_plot = qq_plotly,
      hist_plot = hist_plotly,
      shapiro_pvalue = sw_test$p.value,
      ljung_box_pvalue = lb_test$p.value
    )
  )
}

# 3.d) SNaive forecast analysis
forecast_analysis <- function(mois_ts, start_test_date) {
  start_test_date <- as.Date(start_test_date)
  
  if (format(start_test_date, "%Y-%m") == "2022-05") {
    end_train <- c(2022, 4)
    test_end <- c(2023, 4)
    title_period <- "2022/2023"
  } else if (format(start_test_date, "%Y-%m") == "2023-05") {
    end_train <- c(2023, 4)
    test_end <- c(2024, 4)
    title_period <- "2023/2024"
  } else {
    stop("Invalid start_test_date.")
  }
  
  train_data_snaive <- window(mois_ts, end = end_train)
  test_data_snaive <- window(
    mois_ts,
    start = c(
      as.numeric(format(start_test_date, "%Y")),
      as.numeric(format(start_test_date, "%m"))
    ),
    end = test_end
  )
  
  snaive_forecast <- snaive(
    train_data_snaive,
    h = length(test_data_snaive)
  )
  accuracy_metrics <- forecast::accuracy(
    snaive_forecast,
    test_data_snaive
  )
  
  test_len <- length(test_data_snaive)
  test_dates <- seq(
    start_test_date,
    by = "month",
    length.out = test_len
  )
  
  train_len <- length(train_data_snaive)
  historical_dates <- seq(
    as.Date("2012-01-01"),
    by = "month",
    length.out = train_len
  )
  
  plot_full <- plot_ly() %>%
    add_lines(
      x = historical_dates,
      y = as.numeric(train_data_snaive),
      name = "Historical data"
    ) %>%
    add_lines(
      x = test_dates,
      y = as.numeric(test_data_snaive),
      name = "Actual data (test)",
      line = list(color='red')
    ) %>%
    add_lines(
      x = test_dates,
      y = as.numeric(snaive_forecast$mean),
      name = "SNaive forecast",
      line = list(color='blue', dash='dash')
    ) %>%
    layout(
      title = paste("SNaive forecast (", title_period, ")"),
      xaxis = list(title="Date"),
      yaxis = list(title="Consumption (MW)"),
      hovermode='x unified'
    )
  
  plot_zoom <- plot_full %>%
    layout(
      title = paste("SNaive forecast (Zoom, ", title_period, ")"),
      xaxis = list(range = c(min(test_dates), max(test_dates)))
    )
  
  return(
    list(
      accuracy_metrics = accuracy_metrics,
      plot_full = plot_full,
      plot_zoom = plot_zoom
    )
  )
}

# 3.e) Rolling forecast function (parametric ARIMA)
rolling_forecast <- function(
    h,
    combined_train_refine,
    test_data,
    start_date,
    start_test_date,
    p = 0, d = 1, q = 4,
    P = 3, D = 0, Q = 0,
    period = 12
) {
  
  rolling_forecasts <- ts(
    rep(NA, length(test_data)),
    start = start_test_date,
    frequency = period
  )
  
  n_forecasts <- ceiling(length(test_data) / h)
  dernier_modele <- NULL
  
  for (i in 0:(n_forecasts - 1)) {
    test_start_index <- i * h + 1
    train_end_time <- time(mois_ts)[
      length(combined_train_refine) + test_start_index - 1
    ]
    train_window <- window(
      mois_ts,
      end = train_end_time
    )
    model <- tryCatch(
      {
        Arima(
          train_window,
          order = c(p, d, q),
          seasonal = list(order = c(P, D, Q), period = period),
          method = "ML"
        )
      },
      error = function(e) {
        showNotification(
          "Unable to fit this model. Please change parameters.",
          type = "error"
        )
        return(NULL)
      }
    )
    if (is.null(model)) {
      return(list(
        error = TRUE,
        message = "Model fitting failed."
      ))
    }
    dernier_modele <- model
    
    fc <- forecast(model, h = h)
    forecast_indices <- test_start_index:(test_start_index + h - 1)
    valid_indices <- forecast_indices[forecast_indices <= length(test_data)]
    rolling_forecasts[valid_indices] <- fc$mean[1:length(valid_indices)]
  }
  
  valid_idx <- which(!is.na(rolling_forecasts))
  actuals <- test_data[valid_idx]
  preds <- rolling_forecasts[valid_idx]
  
  mae_ <- mae(actuals, preds)
  rmse_ <- rmse(actuals, preds)
  mape_ <- mape(actuals, preds) * 100
  
  hist_len <- length(combined_train_refine)
  historical_dates <- seq(
    as.Date(paste(start_date[1], start_date[2], "01", sep = "-")),
    by = "month",
    length.out = hist_len
  )
  test_dates <- seq(
    as.Date(paste(start_test_date[1], start_test_date[2], "01", sep = "-")),
    by = "month",
    length.out = length(test_data)
  )
  
  modele_nom <- paste0(
    "SARIMA(", p, ",", d, ",", q, ")(",
    P, ",", D, ",", Q, ")[", period, "]"
  )
  
  plot_full <- plot_ly() %>%
    add_lines(
      x = historical_dates,
      y = as.numeric(combined_train_refine),
      name = "Historical data"
    ) %>%
    add_lines(
      x = test_dates,
      y = as.numeric(test_data),
      name = "Actual data",
      line = list(color = 'red')
    ) %>%
    add_lines(
      x = test_dates,
      y = as.numeric(rolling_forecasts),
      name = "Rolling forecast",
      line = list(color = 'blue', dash = 'dash')
    ) %>%
    layout(
      title = paste("Rolling forecast -", modele_nom, "h=", h),
      xaxis = list(title = "Date"),
      yaxis = list(title = "Consumption (MW)"),
      hovermode = 'x unified'
    )
  
  plot_zoom <- plot_full %>%
    layout(
      title = paste("Zoom on test period -", modele_nom, "h=", h),
      xaxis = list(range = c(min(test_dates), max(test_dates)))
    )
  
  residus_plots <- analyse_residus(
    dernier_modele,
    lag_max = 50,
    title_prefix = modele_nom
  )
  
  return(list(
    error = FALSE,
    forecasts = rolling_forecasts,
    mae = mae_,
    rmse = rmse_,
    mape = mape_,
    plot_full = plot_full,
    plot_zoom = plot_zoom,
    last_model_resid_plots = residus_plots
  ))
}

# -------------------------------------------------------------------------
# 4) Shiny UI
# ------------------------------------------------------------------------- 

ui <- fluidPage(
  theme = shinytheme("yeti"),
  navbarPage(
    "Electricity consumption in France (2012-2024)",
    
    #----------------------------------------- 
    # HOME (exploratory analysis)
    #----------------------------------------- 
    tabPanel(
      "Home",
      
      h2("Visualization of consumption"),
      p("Welcome to the home page of the application. Here, the data will 
        be displayed through exploratory visualizations."),
      hr(),
      hr(),
      
      h3("1) Global consumption (half-hour precision)"),
      plotlyOutput("global_consumption_plot"),
      hr(),
      hr(),
      
      h3("2) Aggregated monthly consumption"),
      plotlyOutput("monthly_consumption_plot"),
      hr(),
      
      h3("3) Monthplot of consumption"),
      plotlyOutput("monthplot_consumption"),
      hr(),
      hr(),
      
      # --- 4) Normalized monthplot (English month names) ---
      h3("4) Normalized monthplot (monthly daily average)"),
      plotlyOutput("plot_monthly_daily_mean"),
      hr(),
      hr(),
      
      # --- 5) Boxplot of monthly daily consumption (English month names) ---
      h3("5) Boxplot of monthly daily consumption"),
      plotlyOutput("boxplot_monthly"),
      hr(),
      hr(),
      
      h3("6) Mean consumption by half-hour"),
      plotlyOutput("mean_consumption_hour"),
      hr(),
      hr(),
      
      h3("7) Boxplot by 30-minute interval"),
      plotlyOutput("boxplot_half_hour"),
      hr(),
      hr(),
      
      h3("8) Seasonal curves (mean) by 30-minute interval"),
      plotlyOutput("plot_hourly_season_mean"),
      hr(),
      hr(),
      
      # --- 9) Mean consumption by day of the week (English day names) ---
      h3("9) Mean consumption by day of the week"),
      plotlyOutput("mean_consumption_day"),
      hr(),
      hr(),
      
      h3("10) STL seasonal decomposition"),
      plotlyOutput("stl_decompose"),
      hr(),
      hr(),
      
      uiOutput("test_results"),
      hr(),
      
      h3("11) Preliminary data analysis"),
      uiOutput("test_preli"),
      p("Data appear to be stationary (based on tests)."),
      fluidRow(
        column(6, plotlyOutput("acf_plot")),
        column(6, plotlyOutput("pacf_plot"))
      ),
      hr(),
      uiOutput("test_D"),
      hr(),
      hr()
    ),
    
    #----------------------------------------- 
    # EXPLORATORY MODELING
    #----------------------------------------- 
    tabPanel(
      "Exploratory modeling",
      sidebarLayout(
        sidebarPanel(
          h4("Choice of exploratory model"),
          radioButtons(
            "mod_type",
            "Model type:",
            choices = c("auto.arima", "Graphically determined SARIMA"),
            selected = "auto.arima"
          ),
          conditionalPanel(
            condition = "input.mod_type == 'Graphically determined SARIMA'",
            br(),
            p("Example: SARIMA(2,0,1)(1,1,1)[12]."),
            actionButton("run_manual_sarima", "Run manual SARIMA model")
          ),
          hr(),
          h5("Diagnostics"),
          verbatimTextOutput("mod_diag_text")
        ),
        mainPanel(
          h3("Overview of historical data and test forecasts"),
          plotlyOutput("mod_fit_plot_full"),
          br(),
          plotlyOutput("mod_fit_plot_zoom"),
          br(),
          
          h4("ACF / PACF / QQPlot / Histogram"), hr(),
          plotlyOutput("mod_resid_acf"), hr(), hr(),
          plotlyOutput("mod_resid_pacf"), hr(), hr(),
          plotlyOutput("mod_resid_qqplot"), hr(), hr(),
          plotlyOutput("mod_resid_hist")
        )
      )
    ),
    
    #-----------------------------------------
    # SNaive
    #----------------------------------------- 
    tabPanel(
      "SNaive",
      fluidRow(
        column(
          4,
          selectInput(
            "snaive_scenario",
            "Choose the forecast period:",
            choices = c(
              "2022/2023"="2022-05-01",
              "2023/2024"="2023-05-01"
            ),
            selected = "2022-05-01"
          ),
          actionButton("run_snaive", "Run SNaive forecast"),
          verbatimTextOutput("snaive_accuracy")
        ),
        column(
          8,
          h3("Full chart"),
          plotlyOutput("snaive_global_plot"),
          br(),
          h3("Zoom on test period"),
          plotlyOutput("snaive_zoom_plot")
        )
      )
    ),
    
    #----------------------------------------- 
    # ARIMA (parameter tuning)
    #-----------------------------------------
    tabPanel(
      "ARIMA",
      sidebarLayout(
        sidebarPanel(
          h4("Parameters SARIMA(p,d,q)(P,D,Q)[12]"),
          numericInput("p", "p:", value=0, min=0, max=10),
          numericInput("d", "d:", value=1, min=0, max=3),
          numericInput("q", "q:", value=4, min=0, max=10),
          numericInput("P", "P:", value=3, min=0, max=10),
          numericInput("D", "D:", value=0, min=0, max=3),
          numericInput("Q", "Q:", value=0, min=0, max=10),
          selectInput(
            "arima_h",
            "Horizon h (months):",
            choices=c(1,2,3,6,12),
            selected=12
          ),
          br(),
          actionButton("run_arima", "Run forecast"),
          hr(),
          verbatimTextOutput("arima_metrics")
        ),
        mainPanel(
          h3("Parametric ARIMA forecasts (rolling forecast)"),
          plotlyOutput("arima_forecast_full"),
          br(),
          plotlyOutput("arima_forecast_zoom"),
          br(),
          h4("Residual diagnostics of the last model"), hr(),
          plotlyOutput("arima_resid_acf"), hr(), hr(),
          plotlyOutput("arima_resid_pacf"), hr(), hr(),
          plotlyOutput("arima_resid_qqplot"), hr(), hr(),
          plotlyOutput("arima_resid_hist")
        )
      )
    )
  ),
  tags$footer(
    tags$p(
      "Developed by Felipe Alvarez Suarez",
      style = "text-align:center; font-size:12px; padding:10px; background-color: #f8f9fa;"
    )
  )
)

# ------------------------------------------------------------------------- 
# 5) Shiny server
# ------------------------------------------------------------------------- 

server <- function(input, output, session) {
  
  # ---------------------------------------------------------------------- 
  # 5.A) HOME : visualizations
  # ---------------------------------------------------------------------- 
  
  # 1) Global consumption (half-hour)
  output$global_consumption_plot <- renderPlotly({
    plot_ly(
      data,
      x = ~Date_Heure,
      y = ~Conso_brute_electricite_MW,
      type = 'scatter',
      mode = 'lines',
      line = list(color='blue', width=2),
      text = ~paste(
        "Date:", Date,
        "<br>Hour:", Heure,
        "<br>Consumption (MW):", Conso_brute_electricite_MW
      ),
      hoverinfo = 'text'
    ) %>%
      layout(
        title="Raw consumption (30-minute intervals)",
        xaxis=list(title="Date", tickangle=45),
        yaxis=list(title="MW"),
        hovermode='x unified'
      )
  })
  
  # 2) Monthly consumption
  output$monthly_consumption_plot <- renderPlotly({
    date_range <- seq(
      as.Date("2012-01-01"),
      by = "month",
      length.out = nrow(data_monthly)
    )
    plot_ly(
      x = ~date_range,
      y = ~data_monthly$Conso_totale_mensuelle_MW,
      type = 'scatter',
      mode = 'lines'
    ) %>%
      layout(
        title = "Monthly raw consumption",
        xaxis = list(title = "Date"),
        yaxis = list(title = "MW")
      )
  })
  
  # 3) Monthplot
  output$monthplot_consumption <- renderPlotly({
    df_mois_ts <- data.frame(
      date = seq(
        from = as.Date("2012-01-01"),
        by = "month",
        length.out = length(mois_ts)
      ),
      conso = as.numeric(mois_ts)
    )
    df_mois_ts$Year <- as.factor(year(df_mois_ts$date))
    df_mois_ts$Month <- month(
      df_mois_ts$date, label = TRUE, abbr = TRUE, locale = "en_US.UTF-8"
    )
    g <- ggplot(
      df_mois_ts,
      aes(x = Month, y = as.numeric(conso), color = Year, group = Year)
    ) +
      geom_line() +
      ggtitle("Monthplot of consumption") +
      xlab("Month") + ylab("Consumption (MW)") +
      scale_y_continuous(labels = scales::comma_format(big.mark = " ", decimal.mark = ",")) +
      theme_minimal()
    ggplotly(g, tooltip = "y")
  })
  
  # 4) Normalized monthplot (monthly daily average) with English month labels
  output$plot_monthly_daily_mean <- renderPlotly({
    # Convert each row's month into numeric (1-12)
    data_monthly$month_num <- as.numeric(
      format(as.Date(paste0(data_monthly$Mois, "-01")), "%m")
    )
    # Compute average daily consumption by month
    moyenne_mensuelle <- data_monthly %>%
      group_by(month_num) %>%
      summarise(daily_mean = round(mean(Conso_moy_jour_MW, na.rm = TRUE)))
    
    # We want English month abbreviations:
    # Jan, Feb, Mar, etc.
    mois_noms <- month(1:12, label = TRUE, abbr = TRUE, locale = "en_US.UTF-8")
    
    plot_ly(
      moyenne_mensuelle,
      x = ~month_num,
      y = ~daily_mean,
      type = 'scatter',
      mode = 'lines+markers',
      line = list(color = 'blue', width = 2),
      marker = list(size = 6, color = 'red')
    ) %>%
      layout(
        title = "Monthly daily average consumption",
        xaxis = list(
          title = "Month", 
          tickmode = "array", 
          tickvals = 1:12, 
          ticktext = mois_noms
        ),
        yaxis = list(title = "Daily consumption (MW)")
      )
  })
  
  # 5) Boxplot of monthly daily consumption with English month names
  output$boxplot_monthly <- renderPlotly({
    # Make sure 'mois_num' or the factor is in English
    # We'll create a version that uses month_num (1-12) to label the boxplot
    temp_df <- data_monthly %>%
      mutate(
        # numeric 1..12
        english_month = as.numeric(format(as.Date(paste0(Mois, "-01")), "%m"))
      )
    
    # Generate abbreviations in English
    english_month_labels <- month(1:12, label = TRUE, abbr = TRUE, locale = "en_US.UTF-8")
    
    # We'll plot using a factor with these English abbreviations
    temp_df$english_month <- factor(
      temp_df$english_month,
      levels = 1:12,
      labels = english_month_labels
    )
    
    plot_ly(
      data = temp_df,
      x = ~english_month,
      y = ~Conso_moy_jour_MW,
      type = 'box'
    ) %>%
      layout(
        xaxis = list(title = "Month"),
        yaxis = list(title = "Daily average consumption (MW)"),
        hoverlabel = list(namelength = -1)
      )
  })
  
  # 6) Mean consumption by half-hour
  output$mean_consumption_hour <- renderPlotly({
    moyenne_par_heure <- data %>%
      group_by(Heure) %>%
      summarise(hourly_mean = mean(
        Conso_brute_electricite_MW, na.rm = TRUE
      ))
    plot_ly(
      moyenne_par_heure,
      x = ~Heure,
      y = ~hourly_mean,
      type = 'scatter',
      mode = 'lines+markers',
      line = list(color = 'blue', width = 2),
      marker = list(size = 6, color = 'red')
    ) %>%
      layout(
        title = "Mean consumption by half-hour",
        xaxis = list(title = "Hour", tickangle = 45),
        yaxis = list(title = "MW")
      )
  })
  
  # 7) Boxplot every 30 minutes
  output$boxplot_half_hour <- renderPlotly({
    plot_ly(
      data,
      x = ~as.factor(Heure),
      y = ~Conso_brute_electricite_MW,
      type = 'box'
    ) %>%
      layout(
        title = "Boxplot of electricity consumption (30-minute intervals)",
        xaxis = list(title = "Hour", tickangle = 45),
        yaxis = list(title = "Consumption (MW)")
      )
  })
  
  # 8) Seasonal curves by half-hour
  output$plot_hourly_season_mean <- renderPlotly({
    data_saison <- data %>%
      mutate(
        # Force month names to English
        mois = month(Date, label = TRUE, abbr = FALSE, locale = "en_US.UTF-8"),
        Heure = format(Date_Heure, "%H:%M")
      ) %>%
      filter(!is.na(Heure)) %>%
      arrange(mois, as.POSIXct(Heure, format = "%H:%M"))
    saison_mean <- data_saison %>%
      group_by(mois, Heure) %>%
      summarise(
        mean_consumption = mean(Conso_brute_electricite_MW, na.rm = TRUE),
        .groups = "drop"
      )
    ggplot_graph <- ggplot(saison_mean) +
      geom_line(
        aes(
          x = Heure,
          y = as.numeric(mean_consumption),
          color = mois,
          group = mois
        )
      ) +
      theme_bw() +
      xlab("Hour") +
      ylab("Mean consumption (MW)") +
      labs(color = "Month") +
      scale_x_discrete(
        breaks = saison_mean$Heure[
          as.numeric(substr(saison_mean$Heure, 4, 5)) == 0
        ]
      ) +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
    
    ggplotly(ggplot_graph, tooltip = c("x", "y", "color")) %>%
      layout(
        hoverlabel = list(bgcolor = "white", font = list(size = 12)),
        yaxis = list(
          title = "Mean consumption (MW)", tickformat = ",.0f"
        )
      )
  })
  
  # 9) Mean consumption by day of the week (English day names)
  output$mean_consumption_day <- renderPlotly({
    # Force day names in English
    data_daily$Jour_semaine <- wday(
      data_daily$Date,
      label = TRUE,
      abbr = FALSE,
      week_start = 1,
      locale = "en_US.UTF-8"  # ensures English day names (Monday, Tuesday, etc.)
    )
    
    mjs <- data_daily %>%
      group_by(Jour_semaine) %>%
      summarise(
        moyenne = mean(Conso_totale_journaliere_MW, na.rm = TRUE)
      )
    
    plot_ly(
      mjs,
      x = ~Jour_semaine,
      y = ~moyenne,
      type = 'scatter',
      mode = 'lines+markers',
      line = list(color = 'blue', width = 2),
      marker = list(size = 6, color = 'red')
    ) %>%
      layout(
        title = "Daily mean by day of the week",
        xaxis = list(title = "Day"),
        yaxis = list(title = "MW")
      )
  })
  
  # 10.a) STL decomposition
  output$stl_decompose <- renderPlotly({
    decomposed <- decompose(train_data)
    
    observed_plot <- plot_ly(
      x = time(decomposed$x),
      y = as.numeric(decomposed$x),
      type = 'scatter', mode = 'lines', name = 'Data'
    )
    trend_plot <- plot_ly(
      x = time(decomposed$trend),
      y = as.numeric(decomposed$trend),
      type = 'scatter', mode = 'lines', name = 'Trend'
    )
    seasonal_plot <- plot_ly(
      x = time(decomposed$seasonal),
      y = as.numeric(decomposed$seasonal),
      type = 'scatter', mode = 'lines', name = 'Seasonality'
    )
    random_plot <- plot_ly(
      x = time(decomposed$random),
      y = as.numeric(decomposed$random),
      type = 'scatter', mode = 'lines', name = 'Residuals'
    )
    
    final_plot <- subplot(
      observed_plot, trend_plot, seasonal_plot, random_plot,
      nrows = 4, shareX = TRUE
    ) %>%
      layout(
        title = list(
          text = "STL decomposition (monthly training data)",
          font = list(size = 16, color = "black"),
          x = 0.5
        )
      )
    final_plot
  })
  
  output$test_results <- renderUI({
    decomposed <- decompose(train_data)
    residus <- na.omit(decomposed$random)
    
    shapiro_result <- shapiro.test(residus)
    ljung_box_result <- Box.test(residus, lag = 50, type = "Ljung-Box")
    
    tagList(
      h4("Statistical tests (STL decomposition)"),
      p(strong("Shapiro-Wilk (normality): "),
        paste0("W = ", round(shapiro_result$statistic, 3), 
               ", p-value = ", format.pval(shapiro_result$p.value))),
      p(strong("Ljung-Box (autocorrelation): "),
        paste0("Chi-sq = ", round(ljung_box_result$statistic, 3), 
               ", p-value = ", format.pval(ljung_box_result$p.value)))
    )
  })
  
  # 11.a) Stationarity tests on training data
  output$test_preli <- renderUI({
    suppressWarnings(
      adf_test <- adf.test(train_data, alternative = "stationary")
    )
    suppressWarnings(
      kpss_test <- kpss.test(train_data, null = "Level")
    )
    
    tagList(
      h4("Stationarity tests (training set up to 2022)"),
      p(strong("ADF test (stationarity): "),
        paste0("Dickey-Fuller = ", round(adf_test$statistic, 3), 
               ", p-value = ", format.pval(adf_test$p.value))),
      p(strong("KPSS test (stationarity): "),
        paste0("KPSS = ", round(kpss_test$statistic, 3), 
               ", p-value = ", format.pval(kpss_test$p.value)))
    )
  })
  
  # 11.b) Preliminary analysis
  
  # 11.b.I) ACF
  output$acf_plot <- renderPlotly({
    acf_result <- acf(train_data, lag.max = 50, plot = FALSE)
    
    acf_data <- data.frame(
      lag = acf_result$lag[-1],
      acf = acf_result$acf[-1],
      ci_upper = 1.96 / sqrt(length(train_data)),
      ci_lower = -1.96 / sqrt(length(train_data))
    )
    
    acf_plot <- ggplot(acf_data, aes(x = lag, y = acf)) +
      geom_bar(stat = "identity", fill = "blue", alpha = 0.7) +
      geom_hline(yintercept = acf_data$ci_upper[1], linetype = "dashed", 
                 color = "red", linewidth = 0.8) +
      geom_hline(yintercept = acf_data$ci_lower[1], linetype = "dashed", 
                 color = "red", linewidth = 0.8) +
      theme_minimal() +
      labs(
        title = "ACF of raw data (monthly)",
        x = "Lags",
        y = "ACF"
      )
    
    ggplotly(acf_plot)
  })
  
  # 11.b.II) PACF
  output$pacf_plot <- renderPlotly({
    pacf_result <- pacf(train_data, lag.max = 50, plot = FALSE)
    
    pacf_data <- data.frame(
      lag = pacf_result$lag[-1],
      pacf = pacf_result$acf[-1],
      ci_upper = 1.96 / sqrt(length(train_data)),
      ci_lower = -1.96 / sqrt(length(train_data))
    )
    
    pacf_plot <- ggplot(pacf_data, aes(x = lag, y = pacf)) +
      geom_bar(stat = "identity", fill = "darkgreen", alpha = 0.7) +
      geom_hline(yintercept = pacf_data$ci_upper[1], linetype = "dashed",
                 color = "red", linewidth = 0.8) +
      geom_hline(yintercept = pacf_data$ci_lower[1], linetype = "dashed",
                 color = "red", linewidth = 0.8) +
      theme_minimal() +
      labs(title = "PACF of raw data (monthly)", x = "Lags", y = "PACF")
    
    ggplotly(pacf_plot)
  })
  
  # 11.c) Tests for differentiation
  output$test_D <- renderUI({
    d <- ndiffs(train_data)
    D <- nsdiffs(train_data)
    
    tagList(
      h4("Differentiation tests"),
      p(strong("Number of non-seasonal differences (d): "), d),
      p(strong("Number of seasonal differences (D): "), D)
    )
  })
  
  # ---------------------------- 
  # 5.B) EXPLORATORY MODELING 
  # ---------------------------- 
  
  auto_arima_model <- reactive({
    mod <- Arima(
      combined_train_refine,
      order=c(1,0,1),
      seasonal=list(order=c(0,1,2), period=12), method="ML"
    )
    mod
  })
  
  manual_sarima_model <- eventReactive(input$run_manual_sarima, {
    mod <- Arima(
      combined_train_refine,
      order=c(2,0,1),
      seasonal=list(order=c(1,1,1), period=12), method="ML"
    )
    mod
  })
  
  output$mod_diag_text <- renderPrint({
    if (input$mod_type == "auto.arima") {
      model <- auto_arima_model()
      prefix_label = "ARIMA(1,0,1)(0,1,2)[12]"
      cat("=== ARIMA(1,0,1)(0,1,2)[12] (auto.arima) ===\n\n")
      print(summary(model))
    } else {
      req(manual_sarima_model())
      model <- manual_sarima_model()
      prefix_label = "SARIMA(2,0,1)(1,1,1)[12]"
      cat("=== SARIMA(2,0,1)(1,1,1)[12] (manual) ===\n\n")
      print(summary(model))
    }
    
    fc <- forecast(model, h = 12)
    preds <- fc$mean[1:12]
    actuals <- test_data[1:12]
    mape_val <- mape(actuals, preds) * 100
    rmse_val <- rmse(actuals, preds)
    
    cat("\n--- Forecast evaluation ---\n")
    cat("RMSE :", round(rmse_val, 3), "\n")
    cat("MAPE :", round(mape_val, 2), "%\n\n")
    
    resid_analysis <- analyse_residus(model, lag_max = 50, title_prefix = prefix_label)
    cat("--- Residuals tests ---\n")
    cat("Shapiro-Wilk p-value :", resid_analysis$shapiro_pvalue, "\n")
    cat("Ljung-Box p-value    :", resid_analysis$ljung_box_pvalue, "\n")
  })
  
  output$mod_fit_plot_full <- renderPlotly({
    if (input$mod_type == "auto.arima") {
      model <- auto_arima_model()
      label <- "ARIMA(1,0,1)(0,1,2)[12]"
    } else {
      model <- manual_sarima_model()
      label <- "SARIMA(2,0,1)(1,1,1)[12]"
    }
    
    fc <- forecast(model, h = length(test_data))
    train_len <- length(combined_train_refine)
    train_dates <- seq(
      as.Date("2012-01-01"),
      by="month",
      length.out=train_len
    )
    
    test_len <- length(test_data)
    test_dates <- seq(
      as.Date("2023-05-01"),
      by="month",
      length.out=test_len
    )
    
    plot_ly() %>%
      add_lines(
        x = train_dates,
        y = as.numeric(combined_train_refine),
        name = "Historical data"
      ) %>%
      add_lines(
        x = test_dates,
        y = as.numeric(test_data),
        name = "Actual data (test)",
        line = list(color='red')
      ) %>%
      add_lines(
        x = test_dates,
        y = as.numeric(fc$mean),
        name = paste("Forecast", label),
        line = list(color='blue', dash='dash')
      ) %>%
      layout(
        title = paste("Forecast -", label),
        xaxis=list(title="Date"),
        yaxis=list(title="Consumption (MW)"),
        hovermode='x unified'
      )
  })
  
  output$mod_fit_plot_zoom <- renderPlotly({
    if (input$mod_type == "auto.arima") {
      model <- auto_arima_model()
      label <- "ARIMA(1,0,1)(0,1,2)[12]"
    } else {
      model <- manual_sarima_model()
      label <- "SARIMA(2,0,1)(1,1,1)[12]"
    }
    
    fc <- forecast(model, h=length(test_data))
    train_len <- length(combined_train_refine)
    train_dates <- seq(
      as.Date("2012-01-01"),
      by="month",
      length.out=train_len
    )
    
    test_len <- length(test_data)
    test_dates <- seq(
      as.Date("2023-05-01"),
      by="month",
      length.out=test_len
    )
    
    plot_ly() %>%
      add_lines(
        x = train_dates,
        y = as.numeric(combined_train_refine),
        name = "Historical data"
      ) %>%
      add_lines(
        x = test_dates,
        y = as.numeric(test_data),
        name = "Actual data (test)",
        line = list(color='red')
      ) %>%
      add_lines(
        x = test_dates,
        y = as.numeric(fc$mean),
        name = paste("Forecast", label),
        line = list(color='blue', dash='dash')
      ) %>%
      layout(
        title = paste("Zoom on test period -", label),
        xaxis = list(range = c(min(test_dates), max(test_dates))),
        yaxis=list(title="Consumption (MW)"),
        hovermode='x unified'
      )
  })
  
  output$mod_resid_acf <- renderPlotly({
    if (input$mod_type == "auto.arima") {
      model <- auto_arima_model()
      prefix <- "ARIMA(1,0,1)(0,1,2)[12]"
    } else {
      model <- manual_sarima_model()
      prefix <- "SARIMA(2,0,1)(1,1,1)[12]"
    }
    residus_analytics <- analyse_residus(model, lag_max=50, title_prefix=prefix)
    residus_analytics$acf_plot
  })
  
  output$mod_resid_pacf <- renderPlotly({
    if (input$mod_type == "auto.arima") {
      model <- auto_arima_model()
      prefix <- "ARIMA(1,0,1)(0,1,2)[12]"
    } else {
      model <- manual_sarima_model()
      prefix <- "SARIMA(2,0,1)(1,1,1)[12]"
    }
    residus_analytics <- analyse_residus(model, lag_max=50, title_prefix=prefix)
    residus_analytics$pacf_plot
  })
  
  output$mod_resid_qqplot <- renderPlotly({
    if (input$mod_type == "auto.arima") {
      model <- auto_arima_model()
      prefix <- "ARIMA(1,0,1)(0,1,2)[12]"
    } else {
      model <- manual_sarima_model()
      prefix <- "SARIMA(2,0,1)(1,1,1)[12]"
    }
    residus_analytics <- analyse_residus(model, lag_max=40, title_prefix=prefix)
    residus_analytics$qq_plot
  })
  
  output$mod_resid_hist <- renderPlotly({
    if (input$mod_type == "auto.arima") {
      model <- auto_arima_model()
      prefix <- "ARIMA(1,0,1)(0,1,2)[12]"
    } else {
      model <- manual_sarima_model()
      prefix <- "SARIMA(2,0,1)(1,1,1)[12]"
    }
    residus_analytics <- analyse_residus(model, lag_max=40, title_prefix=prefix)
    residus_analytics$hist_plot
  })
  
  # ---------------------------- 
  # 5.C) SNaive 
  # ---------------------------- 
  observeEvent(input$run_snaive, {
    res_snaive <- forecast_analysis(mois_ts, input$snaive_scenario)
    
    output$snaive_accuracy <- renderPrint({
      res_snaive$accuracy_metrics
    })
    output$snaive_global_plot <- renderPlotly({
      res_snaive$plot_full
    })
    output$snaive_zoom_plot <- renderPlotly({
      res_snaive$plot_zoom
    })
  })
  
  # ---------------------------- 
  # 5.D) ARIMA (parameter tuning)
  # ---------------------------- 
  observeEvent(input$run_arima, {
    p_ <- input$p
    d_ <- input$d
    q_ <- input$q
    P_ <- input$P
    D_ <- input$D
    Q_ <- input$Q
    h_ <- as.numeric(input$arima_h)
    
    res_arima <- rolling_forecast(
      h = h_,
      combined_train_refine = combined_train_refine,
      test_data = test_data,
      start_date = c(2012, 1),
      start_test_date = c(2023, 5),
      p = p_,
      d = d_,
      q = q_,
      P = P_,
      D = D_,
      Q = Q_,
      period = 12
    )
    
    output$arima_metrics <- renderPrint({
      cat("=== Forecast errors on test period ===\n")
      cat("MAE  :", res_arima$mae, "\n")
      cat("RMSE :", res_arima$rmse, "\n")
      cat("MAPE :", res_arima$mape, "%\n")
      cat("\n=== Normality and independence tests on residuals (last model) ===\n")
      cat("Shapiro-Wilk p-value :", res_arima$last_model_resid_plots$shapiro_pvalue, "\n")
      cat("Ljung-Box p-value    :", res_arima$last_model_resid_plots$ljung_box_pvalue, "\n")
    })
    
    output$arima_forecast_full <- renderPlotly({
      res_arima$plot_full
    })
    output$arima_forecast_zoom <- renderPlotly({
      res_arima$plot_zoom
    })
    
    output$arima_resid_acf <- renderPlotly({
      res_arima$last_model_resid_plots$acf_plot
    })
    output$arima_resid_pacf <- renderPlotly({
      res_arima$last_model_resid_plots$pacf_plot
    })
    output$arima_resid_qqplot <- renderPlotly({
      res_arima$last_model_resid_plots$qq_plot
    })
    output$arima_resid_hist <- renderPlotly({
      res_arima$last_model_resid_plots$hist_plot
    })
  })
}

# ------------------------------------------------------------------------- 
# 6) Run the application
# ------------------------------------------------------------------------- 
shinyApp(ui=ui, server=server)
