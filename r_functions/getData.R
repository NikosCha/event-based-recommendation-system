library(dplyr)
library(geosphere)
library(lubridate)
require(randomForest)

#rsvp files 1 to 17 
for(j in 1:17) {
  # get infos of events
  mergedData = ""
  for(i in 1:24) {
    data = merge(get(paste0("rsvps_",j))[ , c("event_id", "user_id", "response", "created")], get(paste0("events_",i))[ , c("event_id", "time", "utc_offset", "location_id", "fee_price", "group_id")], by = "event_id", all.x =FALSE, all.y = FALSE , sort = TRUE , sufixes = c("rsvp", "events"), no.dups = TRUE, incomparables = NULL)
    mergedData = rbind(mergedData,data)
  } 
  mergedData = merge(mergedData[ , c("response", "time", "utc_offset", "user_id","event_id","created" , "location_id", "fee_price", "group_id")], locations[ , c("location_id","latitude", "longitude", "city")], by = "location_id", all.x =FALSE, all.y = FALSE , sort = TRUE , sufixes = c("data", "location"), no.dups = TRUE, incomparables = NULL)
  
  assign(paste("ds_", j, sep = ""), select(mergedData, -one_of("location_id")))
}

#rsvp files 1 to 17 
for(j in 1:17) {
  # get infos of users
  mergedData = ""
  for(i in 1:7) {
    data = merge(get(paste0("ds_",j)), get(paste0("users_",i))[ , c("user_id", "latitude", "longitude")], by = "user_id", all.x =FALSE, all.y = FALSE , sort = TRUE , sufixes = c("event", "user"), no.dups = TRUE, incomparables = NULL)
    mergedData = rbind(mergedData,data)
  } 
  
  if(mergedData[1,1] == "") 
  {
    mergedData <- mergedData %>% slice(-1)
  }
  #create the ds file
  assign(paste("ds_", j, sep = ""), mergedData)
  data <- get(paste0("ds_",j))  %>%  rowwise() %>% mutate(distance = as.integer(distm(c(as.numeric(longitude.x), as.numeric(latitude.x)), c(as.numeric(longitude.y), as.numeric(latitude.y)), fun = distHaversine)/1000))
  assign(paste("ds_", j, sep = ""), data)
  
  #get day/hour
  data <- get(paste0("ds_",j))  %>%  rowwise() %>% mutate(date = as.POSIXct(as.numeric(time), tz="UTC" , origin="1970-01-01") + as.numeric(utc_offset))
  
  #hour in one column
  data$time <- as.numeric(format(data$date,"%H"))
  
  #day in another column 
  data$weekDay <- wday(data$date)
  
  #if yes then 0 , if no then 1 , if waitlist then 2
  for(i in 1:length(data$response)){
    if(data$response[i] == "yes"){
      data$response[i] = as.numeric(0)
    }else if(data$response[i] == "no") {
      data$response[i] = as.numeric(1)
    }else if(data$response[i] == "waitlist") {
      data$response[i] = as.numeric(2)
    }
  }
  
  data$response = as.numeric(data$response)
  
  assign(paste("ds_", j, sep = ""), select(data,-one_of("date")))
}

#merge files
dataset = ""
for(j in 1:17) {
  dataset = rbind(dataset,get(paste0("ds_",j)))
}
if(dataset[1,1] == "") 
{
  dataset <- dataset %>% slice(-1)
}
dataset$response = as.numeric(dataset$response)
dataset$distance = as.numeric(dataset$distance)
dataset$time = as.numeric(dataset$time)
dataset$weekDay = as.numeric(dataset$weekDay)

