library(car)
library(caret)
library(varhandle)
library(Ecdat)
cancer.df <- read.csv("cancer_reg-cleaning.csv")
# data cleaning
cancer.df$TARGET_deathRate <- as.integer(ifelse(cancer.df$TARGET_deathRate > 200,
                                               1, 0)) # make the target variable binary 
cancer.df$medIncome <- as.integer(ifelse(cancer.df$medIncome  > 45207,
                                                1, 0))
cancer.df$binnedInc<-as.factor(cancer.df$binnedInc) # make binnedInc as a factor
table(cancer.df$binnedInc)#there are ten levels in binnedInc
cancer.df$fbinnedInc <- as.factor(ifelse(cancer.df$binnedInc == '(42724.4, 45201]',  1, 0)) #the only relative binnedInc level
cancer.df<- tidyr::separate(cancer.df,'Geography',into = c("County/City","State"),sep = ",") # slpit column on comma into two columns, "County/City","State"
cancer.df$StateNorthCarolina  <- as.factor(ifelse(cancer.df$State == ' Oklahoma',  1, 0)) #add the relative State levels as dummy variables
cancer.df$StateKentucky <- as.factor(ifelse(cancer.df$State == " Kentucky",  1, 0))
str(cancer.df)#check results

cancer.df <- cancer.df[ , -c(2,4,9,13,14)] #drop the variables that are meaningless and too good to fit)

# partition data
set.seed(10)
train.index <- sample(c(1:dim(cancer.df)[1]), dim(cancer.df)[1]*0.6)  
train.df <- cancer.df[train.index, ]
test.df <- cancer.df[-train.index, ]

#start with all relevant vars
logistic.reg <- glm(TARGET_deathRate ~ ., data = train.df, family = binomial(link = "logit"))
options(scipen=999)
summary(logistic.reg) 

# create logistic model with only relevant variables
logistic.reg <- glm(TARGET_deathRate ~ medIncome+PctHS18_24+PctPublicCoverage+PctOtherRace+StateNorthCarolina+StateKentucky+fbinnedInc, data = train.df, family = binomial(link = "logit")) 
summary(logistic.reg)

# failed to add interaction term
# check polynomial terms
logistic.reg <- glm(formula = TARGET_deathRate ~ medIncome + poly(PctPrivateCoverage,4)+
                      PctPublicCoverage + PctBachDeg25_Over + PctHS18_24 + 
                      StateNorthCarolina + StateKentucky + PctPublicCoverageAlone, family = binomial(link = "logit"), 
                      data = train.df)
summary(logistic.reg) 
# add a polynomial term
logistic.reg <- glm(formula = TARGET_deathRate ~ medIncome + PctUnemployed16_Over +PctPrivateCoverage+
                      PctOtherRace + PctPublicCoverage + PctBachDeg25_Over + PctHS18_24 +I(PctPrivateCoverage^2)+
                      StateNorthCarolina + StateKentucky , family = binomial(link = "logit"), 
                    data = train.df)
summary(logistic.reg)

outlierTest(logistic.reg) #check for outliers
plot(logistic.reg)


# collinearity check with vif
vif(logistic.reg, digits = 3)

#compute predicted probabilities of specific cases
logistic.reg.pred <- predict(logistic.reg, test.df[, -2], type = "response")
# show a few actual and predicted records
data.frame(actual = test.df$TARGET_deathRate[200:205], predicted = logistic.reg.pred[200:205])

# lift chart to see how well we are classifying
library(gains)
gain <- gains(test.df$TARGET_deathRate, logistic.reg.pred, groups=10)
# plot lift chart
plot(c(0,gain$cume.pct.of.total*sum(test.df$TARGET_deathRate))~c(0,gain$cume.obs), 
     xlab="# cases", ylab="Cumulative", main="", type="l")
lines(c(0,sum(test.df$TARGET_deathRate))~c(0, dim(test.df)[1]), lty=2)

# compute deciles and plot decile-wise chart
heights <- gain$mean.resp/mean(test.df$TARGET_deathRate)
midpoints <- barplot(heights, names.arg = gain$depth, ylim = c(0,9), 
                     xlab = "Percentile", ylab = "Mean Response", main = "Decile-wise lift chart")
text(midpoints, heights+0.5, labels=round(heights, 1), cex = 0.8)

# show odds ratio to help interpret the predictor vars
data.frame(odds = exp(coef(logistic.reg))) 

#confusion matrix
logistic.reg.pred <- predict(logistic.reg, test.df[, -2], type = "response")
confusionMatrix(as.factor(ifelse(logistic.reg$fitted > 0.5, 1, 0)), as.factor(train.df[,2]))

#take a look at the final target variable
hist(cancer.df$TARGET_deathRate)
