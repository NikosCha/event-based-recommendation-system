setwd("~/Documents/Thesis/Dataset/meetup_collection_april_14/collection_april_14")

#import users
for(i in 1:7) {
  assign(paste("users_", i, sep = ""), read.csv(sprintf("users_%d.csv", i),header = TRUE, stringsAsFactors = FALSE)) 
}

#import user_tags
user_tags <- read.csv("user_tags.csv",header = TRUE, stringsAsFactors = FALSE)

#import events
for(i in 1:24) {
  assign(paste("events_", i, sep = ""), read.csv(sprintf("events_%d.csv", i),header = TRUE, stringsAsFactors = FALSE))
}

#import categories
categories <- read.csv("categories.csv",header = TRUE, stringsAsFactors = FALSE)

#import tags
tags <- read.csv("tags.csv",header = TRUE, stringsAsFactors = FALSE)

#import group_events
group_events <- read.csv("group_events.csv",header = TRUE, stringsAsFactors = FALSE)

#import group_tags
group_tags <- read.csv("group_tags.csv",header = TRUE, stringsAsFactors = FALSE)

#import group_users
group_users <- read.csv("group_users.csv",header = TRUE, stringsAsFactors = FALSE)

#import groups
groups <- read.csv("groups.csv",header = TRUE, stringsAsFactors = FALSE)

#import locations
locations <- read.csv("locations.csv",header = TRUE, stringsAsFactors = FALSE)

#import rsvps
for(i in 1:17) {
  assign(paste("rsvps_", i, sep = ""), read.csv(sprintf("rsvps_%d.csv", i), header = TRUE, stringsAsFactors = FALSE)) 
}