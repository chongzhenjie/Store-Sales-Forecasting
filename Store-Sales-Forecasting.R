## ----setup, include = F-----------------------------------------------------------
knitr::opts_chunk$set(message = F, warning = F)


## ---------------------------------------------------------------------------------
library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(gridExtra)
library(lubridate)
library(imputeTS)
library(tictoc)
library(forecast)


## ---------------------------------------------------------------------------------
train_df <- read_csv("train.csv.zip")
glimpse(train_df)

test_df <- read_csv("test.csv.zip")
glimpse(test_df)

stores <- read_csv("stores.csv")
glimpse(stores)

features <- read_csv("features.csv.zip")
glimpse(features)


## ---------------------------------------------------------------------------------
checkForNA <- function(data){
  apply(is.na(data), 2, function(col) paste0(round(mean(col) * 100, 2), "%"))
}


## ---------------------------------------------------------------------------------
checkForNA(train_df)
checkForNA(test_df)


## ---------------------------------------------------------------------------------
addUniqueStoreDept <- function(data){
  mutate(data, storeDept = paste0(Store, "_", Dept),
         .before = 1)
}


## ---------------------------------------------------------------------------------
train_df <- addUniqueStoreDept((train_df))
test_df <- addUniqueStoreDept((test_df))

head(train_df)


## ---------------------------------------------------------------------------------
n_distinct(train_df$storeDept)
n_distinct(test_df$storeDept)


## ---------------------------------------------------------------------------------
train_df <- filter(train_df, storeDept %in% unique(test_df$storeDept))

n_distinct(test_df$storeDept) - n_distinct(train_df$storeDept)


## ---------------------------------------------------------------------------------
(storeDeptNoData <- 
  test_df %>%
  filter(!storeDept %in% unique(train_df$storeDept)) %>%
  .$storeDept %>%
  unique())


## ---------------------------------------------------------------------------------
# Add 1 because the first week is not accounted for in the difference

startTrain <- min(train_df$Date)
endTrain <- max(train_df$Date)

startTest <- min(test_df$Date)
endTest <- max(test_df$Date)

(lengthTrain <- difftime(endTrain, startTrain, units = "weeks") + 1)
(lengthTest <- difftime(endTest, startTest, units = "weeks") + 1)


## ---------------------------------------------------------------------------------
obsPerStoreDept <-
  train_df %>%
  count(storeDept) %>%
  arrange(n) %>%
  rename(numObs = n)

unique(obsPerStoreDept$numObs)


## ----fig.height = 3, fig.width = 8------------------------------------------------
obsPerStoreDept %>%
  count(numObs) %>%
  ggplot(aes(numObs, n)) +
  ylab("Frequency") + xlab("Number of Observations") +
  geom_jitter(color = "orangered", alpha = 0.5, height = 100) +  
  geom_vline(xintercept = 143, lty = 2, lwd = 0.5, color = "steelblue")


## ---------------------------------------------------------------------------------
numObs_vs_weeklySales <- train_df %>%
  merge(obsPerStoreDept, by = "storeDept") %>%
  select(Date, storeDept, Weekly_Sales, numObs)


## ---------------------------------------------------------------------------------
numObsLabels <- c("FALSE" = "numObs == 143", "TRUE" = "numObs < 143")

numObs_vs_weeklySales.aes <- function(data, scales = "free_y"){
  data %>%
  ggplot(aes(fill = as.factor(numObs == 143) ,
             color = as.factor(numObs == 143))) +
  theme(legend.position = "none") +
  facet_grid(rows = vars(numObs < 143),
             labeller = as_labeller(numObsLabels),
             scales = scales)
}


## ----fig.width = 8----------------------------------------------------------------
numObs_vs_weeklySales.aes(numObs_vs_weeklySales) +
  geom_density(aes(Weekly_Sales), alpha = 0.5) +
  coord_cartesian(xlim = c(-5000,100000))


## ---------------------------------------------------------------------------------
(holidayWeeks <-
  train_df %>%
  filter(IsHoliday == T) %>%
  .$Date %>%
  unique())

(weekBeforeHolidays <- holidayWeeks - 7)


## ----fig.width = 8----------------------------------------------------------------
numObs_vs_weeklySales.aes(numObs_vs_weeklySales) +
  stat_summary(aes(Date, Weekly_Sales), fun = median, geom = "line", lwd = 1.3) +
  geom_vline(xintercept = holidayWeeks, lty = 2, lwd = 0.1, alpha = 0.3) +
  geom_vline(xintercept = weekBeforeHolidays, lty = 2, lwd = 0.1, alpha = 0.3)


## ---------------------------------------------------------------------------------
numObs_vs_weeklySales.scatter <- function(fn, title){
  numObs_vs_weeklySales %>%
  group_by(storeDept, numObs) %>%
  summarize(Weekly_Sales = fn(Weekly_Sales)) %>%
  numObs_vs_weeklySales.aes(scales = "fixed") +
  geom_jitter(aes(numObs, Weekly_Sales), width = 3, height = 5000, alpha = 0.3) +
  ggtitle(title)
}


## ----warning = F, fig.height = 6--------------------------------------------------
grid.arrange(numObs_vs_weeklySales.scatter(median, "Median"),
             numObs_vs_weeklySales.scatter(mean, "Mean"),
             numObs_vs_weeklySales.scatter(min, "Min"),
             numObs_vs_weeklySales.scatter(max, "Max"),
             numObs_vs_weeklySales.scatter(sd, "Standard Deviation"),
             ncol = 3, nrow = 2)


## ---------------------------------------------------------------------------------
trainDates <- tibble("Date" = seq(startTrain, endTrain, by = 7))

mergeTS <- function(data){
  storeDept <- unique(data$storeDept)
  Store <- unique(data$Store)
  Dept <- unique(data$Dept)
  merge(data, trainDates, by = "Date", all = T) %>%
  replace_na(list(storeDept = storeDept, 
                  Store = Store, 
                  Dept = Dept #, 
                 # Weekly_Sales = 0
                 ))
}


## ---------------------------------------------------------------------------------
storeDept_df <-
  train_df %>%
  select(storeDept, Store, Dept, Date, Weekly_Sales) %>%
  group_by(storeDept) %>%
  do(mergeTS(.)) %>%
  ungroup() %>%
  arrange(Store, Dept)

storeDept_df


## ---------------------------------------------------------------------------------
storeDept_ts<- 
  storeDept_df %>%
  select(-Store, -Dept) %>%
  pivot_wider(names_from = storeDept, values_from = Weekly_Sales) %>%
  select(-Date) %>%
  ts(start = decimal_date(startTrain), frequency = 52)

storeDept_ts[, 1]


## ---------------------------------------------------------------------------------
impute <- function(current_ts){
 if(sum(!is.na(ts)) >= 3){
    na_seadec(current_ts)
 } else if(sum(!is.na(ts)) == 2){
   na_interpolation(current_ts)
 } else{
   na_locf(current_ts)
 }
}


## ---------------------------------------------------------------------------------
for(i in 1:ncol(storeDept_ts)){
  storeDept_ts[, i] <- impute(storeDept_ts[, i])
} 

sum(is.na(storeDept_ts))


## ---------------------------------------------------------------------------------
# change index for different storeDept
baseTS <- storeDept_ts[, 111] 
baseTS_train <- baseTS %>% subset(end = 107)


## ---------------------------------------------------------------------------------
snaive_baseTS <- snaive(baseTS_train, 36)

tslm_baseTS <- tslm(baseTS_train ~ trend + season) %>% forecast(h = 36)

arima_fourier_baseTS <- auto.arima(baseTS_train,seasonal = F, 
                                   xreg = fourier(baseTS_train, K = 3)) %>%
  forecast(xreg = fourier(baseTS_train, K = 3, h = 36), h = 36)

sarima_baseTS <- auto.arima(baseTS_train) %>% forecast(h = 36)

stl_arima_baseTS <- stlf(baseTS_train, method = "arima", 36)

stl_ets_baseTS <- stlf(baseTS_train, method = "ets", 36)


## ---------------------------------------------------------------------------------
forecast_plots <- function(ref, fc_list, model_names){
  plt <- autoplot(ref)
  for(i in 1:length(fc_list)){
    plt <- plt + autolayer(fc_list[[i]], series = model_names[i], PI = F)
  }
  plt <- plt +  
    ylab("Weekly_Sales") +
    guides(color = guide_legend(title = "Forecast"))
  plt
}


## ----fig.width = 9, fig.height = 4------------------------------------------------
forecast_plots(baseTS, 
               list(tslm_baseTS,
                    snaive_baseTS,
                    stl_ets_baseTS),
               c("TSLM",
                 "SNaive",
                 "STL-ETS")
)


## ----fig.width = 9, fig.height = 4------------------------------------------------
forecast_plots(baseTS, 
               list(sarima_baseTS,
                    stl_arima_baseTS,
                    arima_fourier_baseTS),
               c("SARIMA",
                 "STL-ARIMA",
                 "ARIMA-Fourier")
)


## ----fig.width = 9, fig.height = 8------------------------------------------------
arima_fourier_plots <- list()
for(j in 1:6){
  fit <- auto.arima(baseTS_train, xreg = fourier(baseTS_train, K = 6 + 2 * j), seasonal = F) 
  fc <- fit %>%
    forecast(xreg = fourier(baseTS_train, K = 6 + 2 * j, h = 36), h = 36)
  arima_fourier_plots[[j]] <- 
    autoplot(baseTS) + 
    autolayer(fc, PI = F, color = "red") + 
    ylab("Weekly_Sales") + 
    ggtitle(paste("K =", 6 + 2 * j))
}

grid.arrange(arima_fourier_plots[[1]],
             arima_fourier_plots[[2]],
             arima_fourier_plots[[3]],
             arima_fourier_plots[[4]],
             arima_fourier_plots[[5]],
             arima_fourier_plots[[6]],
             ncol = 2)


## ---------------------------------------------------------------------------------
holidayWeights <- train_df %>%
  select(Date, IsHoliday) %>%
  unique() %>%
  .$IsHoliday
holidayWeights <- ifelse(holidayWeights, 5, 1)

totalSize <- nrow(storeDept_ts)
trainSize <- round(0.75 * totalSize)
testSize <- totalSize - trainSize

test_weights <- holidayWeights[(totalSize - testSize + 1):totalSize]
train <- storeDept_ts %>% subset(end = trainSize)
test <- storeDept_ts %>% subset(start = trainSize + 1)


## ---------------------------------------------------------------------------------
wmae <- function(fc){
  # rep() to replicate weights for each storeDept
  weights <- as.vector(rep(test_weights, ncol(fc)))
  
  # as.vector() collapse all columns into one
  MetricsWeighted::mae(as.vector(test), as.vector(fc), weights)
}

model_fc <- function(train, h, model, ...){
  
  tic()
  
  # Initialize forecasts with zeroes
  fc_full <- matrix(0, h, ncol(train))
  
  # Iterate through all storeDept to perform forecasting
  for(i in 1:ncol(train)){
    current_ts <- train[, i]
    fc <- model(current_ts, h, ...)
    fc_full[, i] <- fc
  }
  
  toc()
  
  # Return forecasts
  fc_full
}


## ---------------------------------------------------------------------------------
snaive_ <- function(current_ts, h){
  snaive(current_ts, h = h)$mean
}

tslm_ <- function(current_ts, h){
  tslm(current_ts ~ trend + season) %>%
    forecast( h = h) %>%
    .$mean
}
arima_fourier <- function(current_ts, h, K = K){
  auto.arima(current_ts, xreg = fourier(current_ts, K = K), seasonal = F) %>% 
    forecast(xreg = fourier(current_ts, K = K, h = h), h = h) %>%
    .$mean
}

sarima <- function(current_ts, h){
  auto.arima(current_ts) %>%
    forecast(h = h) %>%
    .$mean
}

stl_ets <- function(current_ts, h){
  stlf(current_ts, method = "ets", opt.crit = 'mae', h = h)$mean
}

stl_arima <- function(current_ts, h){
  stlf(current_ts, method = "arima", h = h)$mean
}


## ---------------------------------------------------------------------------------
snaive_fc <- model_fc(train, testSize, snaive_)
tslm_fc <- model_fc(train, testSize, tslm_)
stl_ets_fc <- model_fc(train, testSize, stl_ets)
stl_arima_fc <- model_fc(train, testSize, stl_arima)
sarima_fc <- model_fc(train, testSize, sarima)
arima_fourier_fc <- model_fc(train, testSize, arima_fourier, K = 12)


## ---------------------------------------------------------------------------------
wmae_summary <- 
  tibble("Model" = c("SNaive (Baseline)", "TSLM",
                     "SARIMA", "ARIMA-Fourier",
                     "STL-ARIMA", "STL-ETS"
                     ),
         "WMAE" = c(wmae(snaive_fc), wmae(tslm_fc),
                    wmae(sarima_fc), wmae(arima_fourier_fc),
                    wmae(stl_arima_fc), wmae(stl_ets_fc)
                    ))

wmae_summary %>% arrange(WMAE)


## ---------------------------------------------------------------------------------
average_fc <- (snaive_fc 
               + tslm_fc 
               + sarima_fc 
               + arima_fourier_fc 
               + stl_arima_fc 
               + stl_ets_fc
               ) / 6

average_weak_fc <- (snaive_fc 
                    + tslm_fc 
                    + sarima_fc 
                    ) / 3

wmae_summary %>% 
  add_row(Model = c("Average of all Models", "Weak Models Average"), 
          WMAE = c(wmae(average_fc), wmae(average_weak_fc))) %>%
  arrange(WMAE)


## ---------------------------------------------------------------------------------
final_snaive_fc <- model_fc(storeDept_ts, lengthTest, snaive_)
final_tslm_fc <- model_fc(storeDept_ts, lengthTest, tslm_)
final_stl_ets_fc <- model_fc(storeDept_ts, lengthTest, stl_ets)
final_stl_arima_fc <- model_fc(storeDept_ts, lengthTest, stl_arima)
final_sarima_fc <- model_fc(storeDept_ts, lengthTest, sarima)
final_arima_fourier_fc <- model_fc(storeDept_ts, lengthTest, arima_fourier, K = 12)


## ---------------------------------------------------------------------------------
adjust_full <- function(fc_full){
  adjust <- function(fc){
  if(2 * fc[9] < fc[8]){
    adj <- fc[8] * (2.5 / 7)
    fc[9] <- fc[9] + adj
    fc[8] <- fc[8] - adj
    }
  fc
  }
  apply(fc_full, 2, adjust)
}


## ---------------------------------------------------------------------------------
final_fc <-(adjust_full(final_snaive_fc)
            + adjust_full(final_tslm_fc)
            + adjust_full(final_arima_fourier_fc)
            + adjust_full(final_sarima_fc)
            + adjust_full(final_stl_ets_fc)
            + adjust_full(final_stl_arima_fc)
            ) / 6


## ---------------------------------------------------------------------------------
storeDept_names <- colnames(storeDept_ts)
colnames(final_fc) <- storeDept_names

testDates <- tibble("Date" = seq(startTest, endTest, by = 7))
final <- 
  cbind(testDates, final_fc) %>% 
  pivot_longer(!Date, names_to = "storeDept", values_to = "Weekly_Sales")

(my_forecasts <-
  test_df %>%
  left_join(final, by = c("storeDept", "Date")) %>% 
  replace_na(list(Weekly_Sales = 0)) %>%
  mutate(Id = paste0(storeDept, "_", Date)) %>%
  select(Id, Weekly_Sales))


## ---------------------------------------------------------------------------------
write_csv(my_forecasts, "my_forecasts.csv")

