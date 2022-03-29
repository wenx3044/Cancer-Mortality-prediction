#Xin Wen
library(caret)
library(class)
#Clustering For Knn 
cancer.df <- read.csv("cancer_reg-cleaning.csv")
# set row names to the utilities column
cancer.df<-cancer.df[,c(-9)]
str(cancer.df)
row.names(cancer.df) <- cancer.df[,12]
cancer.df <- cancer.df[,-12]
d <- dist(cancer.df, method = "euclidean")

# normalize input variables
cancer.df.norm <- sapply(cancer.df, scale)

# add row names: utilities
row.names(cancer.df.norm) <- row.names(cancer.df) # compute normalized distance based on variables Sales and FuelCost

# compute normalized DISTANCE paramneter based on all variables
d.norm <- dist(cancer.df.norm, method = "euclidean") 

# hc 
hc6 <- hclust(d.norm, method = "ward.D")
#plot(hc6, hang = -1, ann = FALSE)

memb6 <- cutree(hc6, k = 13) #ward.D
memb6

cat(memb6)#hc
row.names(cancer.df.norm) <- paste(memb6, ": ", row.names(cancer.df), sep = "")
row.names(cancer.df.norm)
#heatmap
heatmap(as.matrix(cancer.df.norm), Colv = NA, hclustfun = hclust, 
        col=rev(paste("grey",1:99,sep="")))
#k-means
set.seed(2)
km <- kmeans(cancer.df.norm, 13)
# show cluster membership
km$cluster
cat(km$cluster)#k-means

# add new variable into dataframe
cancer.df <- data.frame(cancer.df, memb6)
cancer.df <- data.frame(cancer.df, km$cluster)
#cancer.df <- data.frame(cancer.df, km$cluster,memb6)

#Knn
cancer.df$TARGET_deathRate <- as.factor(ifelse(cancer.df$TARGET_deathRate > 200,
                                               1, 0))
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
                     BirthRate=8, memb6=5) #memb6
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
                     BirthRate=8, km.cluster=5) #km$cluster
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
                     BirthRate=8, memb6=5, km.cluster=5) #memb6, km$cluster

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
