library(caret)
library(class)

cancer.df <- read.csv("cancer_reg-cleaning.csv")
cancer.df$TARGET_deathRate <- as.factor(ifelse(cancer.df$TARGET_deathRate > 200,
                                               1, 0))
cancer.df<- tidyr::separate(cancer.df,'Geography',into = c("County/City","State"),sep = ",") # slpit column on comma into two columns, "County/City","State"
cancer.df<-cancer.df[,c(-9,-13,-14)]
str(cancer.df)
class(cancer.df$TARGET_deathRate)
new.df <- data.frame(avgAnnCount = 270, avgDeathsPerYear = 105,incidenceRate = 454.8,
                     medIncome = 30000, popEst2015=25,povertyPercent=16,studyPerCap=200,
                     MedianAge=40,MedianAgeMale=39,
                     MedianAgeFemale = 52,
                     AvgHouseholdSize=3,PercentMarried=50,PctNoHS18_24=20, 
                     PctHS18_24=30,PctSomeCol18_24=40,PctBachDeg18_24=9,
                     PctHS25_Over=30,PctBachDeg25_Over=15,PctEmployed16_Over=20,PctUnemployed16_Over=10,
                     PctPrivateCoverage=50,PctPrivateCoverageAlone=40,PctEmpPrivCoverage=30,
                     PctPublicCoverage=50,PctPublicCoverageAlone=20,PctWhite=70,
                     PctBlack=20,PctAsian=1,PctOtherRace=9,PctMarriedHouseholds=50,
                     BirthRate=8) # <<<<<<<<< new household


set.seed(500)
train.index <- sample(row.names(cancer.df), 0.7*dim(cancer.df)[1])  
valid.index <- setdiff(row.names(cancer.df), train.index)  
train.df <- cancer.df[train.index, ]
valid.df <- cancer.df[valid.index, ]

# initialize normalized training, validation data, complete data frames to originals
train.norm.df <- train.df
valid.norm.df <- valid.df
cancer.norm.df <- cancer.df

# use preProcess() from the caret package to normalize variables.
norm.values <- preProcess(train.df[, -3], method=c("center", "scale"))
train.norm.df[, -3] <- predict(norm.values, train.df[, -3])
valid.norm.df[, -3] <- predict(norm.values, valid.df[, -3])
cancer.norm.df[, -3] <- predict(norm.values, cancer.df[, -3])  # whole thing
new.norm.df <- predict(norm.values, new.df)

#compute knn
Knn <- knn(train = train.norm.df[, -3], cl = train.norm.df[, 3], test = new.norm.df, 
           k = 3)   

row.names(train.df)[attr(Knn, "Knn.index")]  # high or low

# Initialize a data frame with two columns: k, and accuracy.
accuracy.df <- data.frame(k = seq(1, 14, 1), accuracy = rep(0, 14))

for(i in 1:14) {          # <<<< adjust the bounds to look at particular confusion matrix
  knn.pred <- knn(train = train.norm.df[, -3], cl = train.norm.df[, 3], 
                  test = valid.norm.df[, -3], k = i)
  accuracy.df[i, 2] <- confusionMatrix(knn.pred, factor(valid.norm.df[, 3]))$overall[1] 
}
accuracy.df


for(i in 13:13) {  # <<<< adjust the bounds to look only at confusion matrix with specific k
  knn.pred <- knn(train = train.norm.df[, -3], cl = train.norm.df[, 3], 
                  test = valid.norm.df[, -3], k = i)
  accuracy.df[i, 2] <- confusionMatrix(knn.pred, factor(valid.norm.df[, 3]))$overall[1] 
}

confusionMatrix(knn.pred, factor(valid.norm.df[, 3]))
