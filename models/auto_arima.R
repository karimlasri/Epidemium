# library(xlsx)
# library(tsoutliers)
library(forecast)
# library(tseries)
library(lmtest)

function(df,p,n){
  #### df = dataframe with two columns (year + feature), p = data depth, n = number of years to predict ####

  #Order by year
  df = df[order(df$year),]
  names = colnames(df)
  
  #Create time serie
  start = df$year[1]
  end = tail(df$year, n=1)
  mort = data.frame(row.names = start:end)
  mort$sum = 0
  for (year in start:end){
    if (year %in% df$year){
      mort[as.character(year),] = df[,names[2]][which(df$year == year)]
    } 
    else{
      mort[as.character(year),] = NA
    }
  }
  
  #Create ts
  ts = ts(mort, end = end, frequency = 1)
  
  
  #Fit auto arima to replace missing values
  fit_na = auto.arima(ts)
  
  #Use Kalman filter for missing values
  kr <- KalmanSmooth(ts, fit_na$model)
  id.na <- which(is.na(ts))
  for (i in id.na){
    ts[i] <- fit_na$model$Z %*% kr$smooth[i,]
  }
  
  #Selection of depth
  
  # bool = (gqtest(ts~1)$p.value < 0.10)
  # if (bool == TRUE){
  #   ts<-diff(ts)
  # }
  
  ts_p = tail(ts,p)
  
  
  fit_pred = auto.arima(ts_p, seasonal = TRUE)
  ts_pred = forecast(fit_pred, h = n)$mean
  
  # if (bool == TRUE){
  #   ts_pred <-diffinv(ts_pred,xi = tail(ts,1))
  # }
  
  return(ts_pred)
}