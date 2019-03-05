library(ggplot2)
library(dplyr)
library(gridExtra)
library(reshape2)
library("ggpubr")
library(zoo)
library(fscaret)
require(MASS)
library(randomForest)
library(neuralnet)
library(forecast)
library(boot)
library(plyr)
library(xgboost)
library(readr)
library(stringr)
library(caret)
library(leaps)
library(corrplot)

# Reads in the data files(loading the data)
dengue_features_train <- read.csv("data/dengue_features_train.csv", stringsAsFactors = FALSE)

dengue_features_test <- read.csv("data/dengue_features_test.csv", stringsAsFactors = FALSE)

dengue_labels_train <- read.csv("data/dengue_labels_train.csv", stringsAsFactors = FALSE)

submissions <- read.csv("data/submission_format.csv", stringsAsFactors = FALSE)
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
  
  
  

#Importance
# control <- trainControl(method="repeatedcv", number=10, repeats=3)
# # train the model
# model <- train(total_cases~., data=sj_joined, method="lvq", preProcess="scale", trControl=control)
# # estimate variable importance
# importance <- varImp(model, scale=FALSE)
# # summarize importance
# print(importance)
# # plot importance
# plot(importance)

#summary(m1 <- glm.nb(total_cases ~ "binomial", data = sj_joined))



#################################################################################
# MACHINE LEARNING ALGORITHMS - SUPERVISED LEARNING #############################
#################################################################################


##### Neural Networks - Random Forest

sj_rf_model <- randomForest(total_cases~., data = sj_joined)

#print(sj_rf_model)

#sj_rf_predictions <- predict(object = sj_rf_model, dengue_features_test_SJ)

#sj_rf_solution <- data.frame(submissions[1:260,-4], total_cases = round(sj_rf_predictions))

iq_rf_model <- randomForest(total_cases~., data = iq_joined)

#print(iq_rf_model)

#iq_rf_predictions <- predict(object = iq_rf_model, dengue_features_test_IQ)

#iq_rf_solution <- data.frame(submissions[261:416,-4], total_cases =round(iq_rf_predictions))

#rf_solution <- bind_rows(sj_rf_solution,iq_rf_solution)

#write.csv(rf_solution, file = 'dengue_solution_1.csv', row.names = F)


##### Negative Binomial Model

negBinomModel_sj <- glm.nb(total_cases ~ ., data = sj_joined)
summary(negBinomModel_sj)
nb_predictions_sj <- predict(negBinomModel_sj, dengue_features_test_SJ , type="response") 

negBinomModel_iq <- glm.nb(total_cases~., data = iq_joined)
summary(negBinomModel_iq)
nb_predictions_iq <- predict(negBinomModel_iq, dengue_features_test_IQ , type="response")

sj_nb_solution <- data.frame(submissions[1:260,-4], total_cases = round(nb_predictions_sj))

iq_nb_solution <- data.frame(submissions[261:416,-4], total_cases = round(nb_predictions_iq))

nb_solutions <- bind_rows(sj_nb_solution, iq_nb_solution)

#write.csv(nb_solutions, file = "dengue_solution_2.csv", row.names = F)


###### Poisson Regression Model

poissonModel_sj <- glm(total_cases~., family="poisson", data=sj_joined)
summary(poissonModel_sj)
p_predictions_sj <- predict(poissonModel_sj, dengue_features_test_SJ , type="response") 

poissonModel_iq <- glm(total_cases~., family="poisson", data=iq_joined)
summary(poissonModel_iq)
p_predictions_iq <- predict(poissonModel_iq, dengue_features_test_IQ , type="response") 

sj_p_solution <- data.frame(submissions[1:260,-4], total_cases = round(p_predictions_sj))

iq_p_solution <- data.frame(submissions[261:416,-4], total_cases = round(p_predictions_iq))

p_solutions <- bind_rows(sj_p_solution, iq_p_solution)
#write.csv(p_solutions, file = "dengue_solution_3.csv", row.names = F)


##### Gradient Boosting Algorithm
#### XGBoost

# Fitting Model

#TrainControl <- trainControl( method = "repeatedcv", number = 10, repeats = 4)
#model_sj <- train(total_cases~., data = sj_joined, method = "xgbTree", trControl = TrainControl,verbose = FALSE)
#predicted_sj <- predict(model_sj, dengue_features_test_SJ)

#model_iq <- train(total_cases~., data = iq_joined, method = "xgbTree", trControl = TrainControl,verbose = FALSE)

#predicted_iq <- predict(model_iq, dengue_features_test_IQ)

#sj_xgb_solution <- data.frame(submissions[1:260,-4], total_cases = round(predicted_sj))

#iq_xgb_solution <- data.frame(submissions[261:416,-4], total_cases = round(predicted_iq))

#xgb_solutions <- bind_rows(sj_xgb_solution, iq_xgb_solution)
#write.csv(xgb_solutions, file = "dengue_solution_4.csv", row.names = F)


#################################################################################
# Accuracy Score for Each Model #################################################
#################################################################################

# Splitting Data Set

set.seed(10)
trainIndex = createDataPartition(sj_joined$total_cases,
                                 p=0.75, list=FALSE,times=1)
train_data_sj <- sj_joined[trainIndex,]
test_data_sj <- sj_joined[-trainIndex,]


trainIndex = createDataPartition(iq_joined$total_cases,
                                 p=0.75, list=FALSE,times=1)
train_data_iq <- iq_joined[trainIndex,]
test_data_iq <- iq_joined[-trainIndex,]

###### Poisson Regression Model 

poisson_model_train_sj <- glm(total_cases~., family="poisson", data=train_data_sj)

poisson_validation_sj <- predict(poisson_model_train_sj, test_data_sj[,-3],type="response")

poisson_validation_sj <- round(poisson_validation_sj)

test_data_sj["predicted_values"] <- poisson_validation_sj
accuracy_score_poisson_sj <- 1000*sum(poisson_validation_sj == test_data_sj$total_cases)/nrow(test_data_sj)

poisson_model_train_iq <- glm(total_cases~., family="poisson", data=train_data_iq)

poisson_validation_iq <- predict(poisson_model_train_iq, test_data_iq[,-3],type="response")

poisson_validation_iq <- round(poisson_validation_iq)

test_data_iq["predicted_values"] <- poisson_validation_iq

accuracy_score_poisson_iq <- 1000*sum(poisson_validation_iq == test_data_iq$total_cases)/nrow(test_data_iq)




model.full <- regsubsets(total_cases~., data = iq_joined, nbest = 20)
# Collapse predictor names into a single string
predictors <- summary(model.full)$which
fxn <- function(x) {
  paste(names(which(x)), collapse = " ")
}


preds <- apply(predictors, 1, fxn)
npred <- rowSums(predictors)-1
rss <- summary(model.full)$rss

# Add intercept-only model (which has zero R^2)
rsq <- summary(model.full)$rsq
plot.df <- data.frame(preds, npred, rss, rsq, stringsAsFactors = FALSE)
tplot.df <- data.frame(preds, npred, rss, rsq, stringsAsFactors = FALSE)
plot.df <- rbind(plot.df,data.frame(preds = "(Intercept)", npred = 0, rss, rsq = 0))
plot.df = plot.df[order(plot.df$npred),]

# Save predictors for best models at each number of predictors
best.preds = NULL
for(i in 0:6) {
  dfi = plot.df[plot.df$npred==i,]
  best.preds =c(best.preds, dfi[dfi$rss== min(dfi$rss),]$preds)
}

# Indicate rows with best models
plot.df$best = 0
for(i in 1:nrow(plot.df)) {
  if(plot.df[i,]$preds%in%best.preds) {
    plot.df[i,]$best = 1
  }
}

#plot figures
par(mfrow=c(1,2))
with(plot.df,plot(npred, rss))
with(plot.df[plot.df$best==1,],points(npred, rss, pch=19, col="red", type="o"))
with(plot.df,plot(npred, rsq))
with(plot.df[plot.df$best==1,],points(npred, rsq, pch=19, col="red", type="o"))





##### Negative Binomial Regression Model

negBinomModel_sj_train <- glm.nb(total_cases ~ ., data = train_data_sj)
negBinomModel_iq_train <- glm.nb(total_cases~., data = train_data_iq)

nb_validation_sj <- predict(negBinomModel_sj_train, test_data_sj, type = "response")
nb_validation_iq <- predict(negBinomModel_iq_train, test_data_iq, type = "response")

test_data_sj["predicted_values_nb"] <- nb_validation_sj
test_data_iq["predicted_values_nb"] <- nb_validation_iq

accuracy_score_nb <- (sum(test_data_sj$predicted_values_nb == test_data_sj$total_cases)+
                        sum(test_data_iq$predicted_values_nb == test_data_iq$total_cases))/(nrow(test_data_sj)+nrow(test_data_iq)) 



npred = rowSums(predictors) - 1 # number of predictors
cp = summary(model.full)$cp # Cp for each fitted model
bic = summary(model.full)$bic
adjr2 = summary(model.full)$adjr2
plot.df2 = data.frame(preds, npred, cp, bic, adjr2, stringsAsFactors = FALSE)

plot.df2$best = 0
for (i in 1:nrow(plot.df2)) {
  
  if (plot.df2[i,]$preds %in% best.preds) {
    
    plot.df2[i,]$best = 1
    
  } 
}
plot.df2 = plot.df2[plot.df2$best == 1,]


# Plot figures
par(mfrow=c(1,3)) 
with(plot.df2, plot(npred, cp, pch=19, type="o")) 
with(plot.df2, plot(npred, bic, pch=19, type="o")) 
with(plot.df2, plot(npred, adjr2, pch=19, type="o"))



################ Predicted Vs Outcome ###############


## Rain Forest
prediction_train_sj <-  predict(object = sj_rf_model, sj_joined)
prediction_train_iq <-  predict(object = iq_rf_model, iq_joined)

sj_joined["predictions_rf_sj"] <- round(prediction_train_sj)

iq_joined["predictions_rf_sj"] <- round(prediction_train_iq)


rf_sf_pred <- dplyr::select(sj_joined, year, weekofyear, total_cases, predictions_rf_sj)
rf_sf_pred_plot <- ggplot(data = rf_sf_pred) +
  geom_line(mapping = aes(x = weekofyear, y = total_cases, color = "red")) +
  geom_line(mapping = aes(x = weekofyear, y = predictions_rf_sj, color = "blues9")) +
  scale_color_discrete(name = "Total cases type", labels = c("Actual", "Prediction")) +
  facet_wrap(~year) +
  labs(
    title = "San Juan - cases of DF over the years", 
    x = "Week Of The Year", 
    y = "Total cases of DF",
    color = NA
  )

rf_iq_pred <- dplyr::select(iq_joined, year, weekofyear, total_cases, predictions_rf_sj)
rf_iq_pred_plot <- ggplot(data = rf_sf_pred) +
  geom_line(mapping = aes(x = weekofyear, y = total_cases, color = "red")) +
  geom_line(mapping = aes(x = weekofyear, y = predictions_rf_sj, color = "blues9")) +
  scale_color_discrete(name = "Total cases type", labels = c("Actual", "Prediction")) +
  facet_wrap(~year) +
  labs(
    title = "San Juan - cases of DF over the years", 
    x = "Week Of The Year", 
    y = "Total cases of DF",
    color = NA
  )


#########
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
