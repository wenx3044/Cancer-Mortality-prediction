#Xin Wen
library(car)
library(caret)

#Clustering For Logistic Regression
#prepare for the dataframe
cancer.df<- read.csv('cancer_reg-cleaning.csv')
cancer.df$binnedInc<-as.factor(cancer.df$binnedInc) # make binnedInc as a factor
cancer.df$fbinnedInc <- as.integer(ifelse(cancer.df$binnedInc == '(42724.4, 45201]',  1, 0))
cancer.df$Geography_ = cancer.df$Geography 
cancer.df<- tidyr::separate(cancer.df,'Geography_',into = c("County/City","State"),sep = ",") # slpit column on comma into two columns, "County/City","State"
cancer.df$StateNorthCarolina <- as.integer(ifelse(cancer.df$State == ' Oklahoma',  1, 0)) #add the relative State levels as dummy variables
cancer.df$StateKentucky <- as.integer(ifelse(cancer.df$State == " Kentucky",  1, 0))
cancer.df<-cancer.df[,c(-9,-36,-37)]
str(cancer.df)
#Set row names
row.names(cancer.df) <- cancer.df[,12]
cancer.df <- cancer.df[,-12]
d <- dist(cancer.df, method = "euclidean")

cancer.df.norm <- sapply(cancer.df, scale)
row.names(cancer.df.norm) <- row.names(cancer.df) 
d.norm <- dist(cancer.df.norm, method = "euclidean") 

hc6 <- hclust(d.norm, method = "ward.D")#best
#plot(hc6, hang = -1, ann = FALSE)

memb6 <- cutree(hc6, k = 13)
memb6

cat(memb6)#hc
#hist(memb6)#plot

row.names(cancer.df.norm) <- paste(memb6, ": ", row.names(cancer.df), sep = "")
row.names(cancer.df.norm)

#heatmap(as.matrix(cancer.df.norm), Colv = NA, hclustfun = hclust, 
#        col=rev(paste("grey",1:99,sep="")))

set.seed(2)
km <- kmeans(cancer.df.norm, 13)  #k=13 got from last analysis
km$cluster
cat(km$cluster)#k-means
#hist(km$cluster)

# Compare methods
#table(km$cluster, memb6)

# add new variable into dataframe
#cancer1.df <- data.frame(cancer.df, km$cluster)
#cancer2.df <- data.frame(cancer.df, km$cluster)

#cancer3.df <- data.frame(cancer.df, memb6)
#cancer3.df <- data.frame(cancer.df, km$cluster)
cancer3.df <- data.frame(cancer.df, km$cluster,memb6)

#logistic
cancer3.df$TARGET_deathRate <- as.integer(ifelse(cancer3.df$TARGET_deathRate > 200,
                                                 1, 0))
row_names_df_to_remove<-c("Kearny County, Kansas","New Kent County, Virgins","De Baca County, New Mexico")
cancer3.df[!(row.names(cancer3.df) %in% row_names_df_to_remove),]


set.seed(10)
train.index <- sample(c(1:dim(cancer3.df )[1]), dim(cancer3.df )[1]*0.6)  
train.df <- cancer3.df [train.index, ]
test.df <- cancer3.df[-train.index, ]
logistic.reg <- glm(formula = TARGET_deathRate ~ medIncome + PctUnemployed16_Over +PctPrivateCoverage+
                      PctOtherRace + PctPublicCoverage + PctBachDeg25_Over + PctHS18_24 +I(PctPrivateCoverage^2)+
                      StateNorthCarolina + StateKentucky + km.cluster, family = binomial(link = "logit"), 
                    data = train.df)

logistic.reg <- glm(formula = TARGET_deathRate ~ medIncome + PctUnemployed16_Over +PctPrivateCoverage+
                      PctOtherRace + PctPublicCoverage + PctBachDeg25_Over + PctHS18_24 +I(PctPrivateCoverage^2)+
                      StateNorthCarolina + StateKentucky + memb6, family = binomial(link = "logit"), 
                    data = train.df)
logistic.reg <- glm(formula = TARGET_deathRate ~ medIncome + PctUnemployed16_Over +PctPrivateCoverage+
                      PctOtherRace + PctPublicCoverage + PctBachDeg25_Over + PctHS18_24 +I(PctPrivateCoverage^2)+
                      StateNorthCarolina + StateKentucky + memb6 + km.cluster, family = binomial(link = "logit"), 
                    data = train.df)
summary(logistic.reg)

#check outliers
plot(logistic.reg)

#check overfitting
pred <-predict(logistic.reg, test.df, type = "response")
y_pred_num <- ifelse(pred > 0.5, 1, 0)
y_pred <- factor(y_pred_num, levels=c(0, 1))
confusionMatrix(as.factor(y_pred_num), as.factor(test.df[,3]))

# collinearity check with vif
vif(logistic.reg, digits = 3)


#compute predicted probabilities of specific cases
logistic.reg.pred <- predict(logistic.reg, test.df[, -3], type = "response")
# show a few actual and predicted records
data.frame(actual = test.df$TARGET_deathRate[200:205], predicted = logistic.reg.pred[200:205])

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
confusionMatrix(as.factor(ifelse(logistic.reg$fitted > 0.5, 1, 0)), as.factor(train.df[,3]))

#take a look at the final target variable
hist(cancer.df$TARGET_deathRate)

