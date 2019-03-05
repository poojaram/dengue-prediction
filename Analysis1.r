library(ggplot2)
library(dplyr)
library(gridExtra)
library(reshape2)
library("ggpubr")
library(zoo)
library(fscaret)
require(MASS)
library(corrplot)


# Reads in the data files(loading the data)
dengue_features_train <- read.csv("data/dengue_features_train.csv", stringsAsFactors = FALSE)

dengue_features_test <- read.csv("data/dengue_features_test.csv", stringsAsFactors = FALSE)

dengue_labels_train <- read.csv("data/dengue_labels_train.csv", stringsAsFactors = FALSE)

#na_count <- data.frame(colSums(is.na(dengue_features_train)))


# Setting all NA values to the previous ones
dengue_features_train <- dengue_features_train %>%
  do(na.locf(.))
dengue_labels_train <- dengue_labels_train %>%
  do(na.locf(.))

na_count <- data.frame(colSums(is.na(dengue_features_train)))

# Filter of data by city  
sj_labels_train <- filter(dengue_labels_train, city == 'sj')
sj_features_train <- filter(dengue_features_train, city == 'sj')

iq_labels_train <- filter(dengue_labels_train, city == 'iq')
iq_features_train <- filter(dengue_features_train, city == 'iq')

# Joining the labels and features data frames
# Mutating coumns for 4 weeks and 8 weeks of lag
sj_joined <- left_join(sj_labels_train, sj_features_train, by = c("city", "year", "weekofyear")) %>%
  mutate(four_lag_cases = lag(total_cases, 4), eight_lag_cases = lag(total_cases, 8)) %>%
  do(na.locf(.))
# for(i in 1:ncol(sj_joined)){ # replacing NA with mean values
#   sj_joined[is.na(sj_joined[,i]), i] <- mean(sj_joined[,i], na.rm = TRUE)
# }

iq_joined <- left_join(iq_labels_train, iq_features_train, by = c("city", "year", "weekofyear")) %>%
  mutate(four_lag_cases = lag(total_cases, 4), eight_lag_cases = lag(total_cases, 8)) %>%
  do(na.locf(.))
# for(i in 1:ncol(iq_joined)){
#   iq_joined[is.na(iq_joined[,i]), i] <- mean(iq_joined[,i], na.rm = TRUE) 
# }

# Normalised data sets for IQ and SJ
# sj_scaled <- select(sj_joined, -city, -year, -total_cases, -week_start_date, -weekofyear)
# sj_scaled <- as.data.frame( scale(sj_scaled))
# 
# iq_scaled <- select(iq_joined, -city, -year, -total_cases, -week_start_date, -weekofyear)
# iq_scaled <- as.data.frame( scale(iq_scaled))

# Feature selection


# Selcting outcome variable
outcomeName <- c('total_cases')
predictorsNames <- names(sj_joined)[names(sj_joined) != outcomeName]


# Splitting into training and testing data
set.seed(1234)
splitIndex <- createDataPartition(sj_joined[,outcomeName], p = .75, list = FALSE, times = 1)
trainDF <- sj_joined[ splitIndex,]
testDF  <- sj_joined[-splitIndex,]


#  Removing features with low variance
variances<-apply(sj_joined, 2, var)
variances[which(variances<=0.0025)]
names(sj_joined)[nearZeroVar(sj_joined)]

names(iq_joined)[nearZeroVar(iq_joined)]

# Remove redundant features
set.seed(7)

# load the library
library(mlbench)
library(caret)
# load the data
# # Removes station data
sj_joined <- dplyr::select(sj_joined, -city, -week_start_date, -station_max_temp_c, -station_min_temp_c, -station_avg_temp_c, -station_precip_mm, -station_diur_temp_rng_c)
iq_joined <- dplyr::select(iq_joined, -city, -week_start_date, -station_max_temp_c, -station_min_temp_c, -station_avg_temp_c, -station_precip_mm, -station_diur_temp_rng_c)

correlationMatrix <- cor(sj_joined)
# summarize the correlation matrix
print(correlationMatrix)
# find attributes that are highly corrected (ideally >0.75)
highlyCorrelatedSJ <- findCorrelation(correlationMatrix, cutoff=0.75)
# print indexes of highly correlated attributes
print(highlyCorrelatedSJ)

correlationMatrix <- cor(iq_joined)
# summarize the correlation matrix
print(correlationMatrix)
# find attributes that are highly corrected (ideally >0.75)
highlyCorrelatedIQ <- findCorrelation(correlationMatrix, cutoff=0.75)
# print indexes of highly correlated attributes
print(highlyCorrelatedIQ)

dew_vs_hum_iq <- ggplot(data = iq_joined) +
  geom_point(mapping = aes(x = reanalysis_specific_humidity_g_per_kg, y = reanalysis_dew_point_temp_k)) +
  labs(
    title = "Iquitos - Humidity vs Dew Point Temp", # plot title
    x = "Mean specific humidity in G per KG",
    y = "Mean dew point temperature in K",
    color = NA
  )

dew_vs_hum_sj <- ggplot(data = sj_joined) +
  geom_point(mapping = aes(x = reanalysis_specific_humidity_g_per_kg, y = reanalysis_dew_point_temp_k)) +
  labs(
    title = "San Juan - Humidity vs Dew Point Temp", # plot title
    x = "Mean specific humidity in G per KG", 
    y = "Mean dew point temperature in K",
    color = NA
  )

dew_vs_hum <- grid.arrange(dew_vs_hum_sj, dew_vs_hum_iq, ncol = 2)

temp_vs_hum <- ggplot(data = sj_joined) +
  geom_point(mapping = aes(x = four_lag_cases, y = ndvi_se))


# Heat Index mutated into SJ and IQ joines
sj_joined <- mutate(sj_joined, heat_index = reanalysis_dew_point_temp_k + 61 + ((reanalysis_dew_point_temp_k - 68) * 1.2) + reanalysis_specific_humidity_g_per_kg * 0.904) %>%
  mutate(ndvi_mean = ((ndvi_ne + ndvi_nw + ndvi_se + ndvi_sw) / 4)) %>%
  dplyr::select(year, weekofyear, total_cases, heat_index, reanalysis_relative_humidity_percent, reanalysis_max_air_temp_k, ndvi_mean, reanalysis_precip_amt_kg_per_m2)

iq_joined <- mutate(iq_joined, heat_index = reanalysis_dew_point_temp_k + 61 + ((reanalysis_dew_point_temp_k - 68) * 1.2) + reanalysis_specific_humidity_g_per_kg * 0.904)
iq_joined <-  dplyr::select(iq_joined, year, weekofyear, total_cases, heat_index, reanalysis_precip_amt_kg_per_m2, reanalysis_min_air_temp_k, reanalysis_air_temp_k, reanalysis_relative_humidity_percent)

### TESTING DATA

dengue_features_test <-  dengue_features_test %>%
  do(na.locf(.)) %>%
  mutate(heat_index = reanalysis_dew_point_temp_k + 61 + ((reanalysis_dew_point_temp_k - 68) * 1.2) + reanalysis_specific_humidity_g_per_kg * 0.904) %>%
  mutate(ndvi_mean = ((ndvi_ne + ndvi_nw + ndvi_se + ndvi_sw) / 4)) 

dengue_features_test_SJ <- filter(dengue_features_test, city == "sj") %>%
  dplyr::select(year, weekofyear, heat_index, reanalysis_relative_humidity_percent, reanalysis_max_air_temp_k, ndvi_mean, reanalysis_precip_amt_kg_per_m2)

dengue_features_test_IQ <- filter(dengue_features_test, city == "iq") %>%
  dplyr::select(year, weekofyear, heat_index, reanalysis_precip_amt_kg_per_m2, reanalysis_min_air_temp_k, reanalysis_air_temp_k, reanalysis_relative_humidity_percent)
  


  
sj_train_labels <- filter(dengue_labels_train, city == 'sj')
sj_train_features <- filter(dengue_features_train, city == 'sj')
iq_train_labels <- filter(dengue_labels_train, city == 'iq')
iq_train_features <- filter(dengue_features_train, city == 'iq')

# Add total_cases column to *_train_features dataframes
 sj_train_features <- left_join(sj_train_features, sj_train_labels, by = c('city', 'year', 'weekofyear'))
sj_train_features$total_cases <- sj_train_labels$total_cases

 iq_train_features <- left_join(iq_train_features, iq_train_labels, by = c('city', 'year', 'weekofyear'))
iq_train_features$total_cases <- iq_train_labels$total_cases


# Correlation matrix
m_sj_train_features <- data.matrix(sj_train_features)
m_sj_train_features <- cor(x = m_sj_train_features[,3:24], use = 'everything', method = 'pearson')
m_iq_train_features <- data.matrix(iq_train_features)
m_iq_train_features <- cor(x = m_iq_train_features[,3:24], use = 'everything', method = 'pearson')

m_sj_joined_train_features <- data.matrix(sj_joined)
m_sj_joined_train_features <- cor(x = m_sj_joined_train_features, use = 'everything', method = 'pearson')

m_iq_joined_train_features <- data.matrix(iq_joined)
m_iq_joined_train_features <- cor(x = m_iq_joined_train_features, use = 'everything', method = 'pearson')

