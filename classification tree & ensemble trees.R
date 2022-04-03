library(rpart)
library(rpart.plot)
#read the dataset
cancer.df <- read.csv('cancer_reg-cleaning.csv')
# make the target variable binary 
cancer.df$TARGET_deathRate <- as.integer(ifelse(cancer.df$TARGET_deathRate > 200,
                                                1, 0)) 
# make binnedInc as a factor
cancer.df$binnedInc<-as.factor(cancer.df$binnedInc) 
# slpit column on comma into two columns, "County/City","State"
cancer.df<- tidyr::separate(cancer.df,'Geography',into = c("County/City","State"),sep = ",") 
#remove the Geography
cancer.df <-cancer.df[ , -c(13)]
#make State as factor
cancer.df$State<-as.factor(cancer.df$State) 


#validation
set.seed(2)  
train.index <- sample(c(1:dim(cancer.df)[1]), dim(cancer.df)[1]*0.6)  
train.df <- cancer.df[train.index, ]
valid.df <- cancer.df[-train.index, ]

#control by depth
default.ct <- rpart(TARGET_deathRate ~ ., data = train.df, method = "class",   
                    control = rpart.control(maxdepth = 5))

#control by cp (cost of complexity penalty)
deeper.ct <- rpart(TARGET_deathRate ~ ., data = train.df, method = "class", 
                   cp = 0.03, minsplit = 1) 


library(caret)

#use grid search to find the best tree
curr_F1 <- 0 
best_cost_penalty <- 0
best_min_leaf_to_split <- 2

for( cost_penalty in seq(from=0, to=0.01, by=0.001)) {
  for( min_leaf_to_split in seq(from=10, to=20, by=1)) {
    
    # train the tree
    trained_tree <- rpart(TARGET_deathRate ~ ., data = train.df, method = "class", 
                          cp = cost_penalty, minsplit = min_leaf_to_split)
    
    # predict with the trained tree
    train.results <- predict( trained_tree, train.df, type = "class" )
    valid.results <- predict( trained_tree, valid.df, type = "class" )  
    
    # generate the confusion matrix to compare the prediction with the actual value of TARGET_deathRate (0/1), 
    # to calculate the sensitivity and specificity
    results <- confusionMatrix( valid.results, as.factor(valid.df$TARGET_deathRate) )
    
    # calculate F1 from results
    Sensitivity <- results$byClass[1] 
    Specificity <- results$byClass[2] 
    F1 <- (2 * Sensitivity * Specificity) / (Sensitivity + Specificity)
    
    # If this is the best F1 we have so far, store the current values:
    if( F1 > curr_F1 ) {
      curr_F1 <- F1
      best_cost_penalty <- cost_penalty
      best_min_leaf_to_split <- min_leaf_to_split
    }
  }
}
cat("best F1=" , curr_F1, "; best best_cost_penalty=", best_cost_penalty, 
    "; best_min_leaf_to_split=", best_min_leaf_to_split)

# retrain the tree to match the best parameters we found  
trained_tree <- rpart(TARGET_deathRate ~ ., data = train.df, method = "class", 
                      cp = best_cost_penalty , minsplit = best_min_leaf_to_split ) 
# plot that best tree 
prp(trained_tree, type = 1, extra = 1, under = TRUE, split.font = 1, varlen = -10, 
    box.col=ifelse(trained_tree$frame$var == "<leaf>", 'gray', 'white'))  
results
confusionMatrix( train.results, as.factor(train.df$TARGET_deathRate) )



# use cp to get the optimal tree size
cv.ct <- rpart(TARGET_deathRate ~ ., data = train.df, method = "class", 
               cp = 0.009, minsplit = 1, xval = 5)   #5 folds
# print the cp table  
printcp(cv.ct)
# prune by minimum cp
pruned.ct <- prune(cv.ct, 
                   cp = cv.ct$cptable[which.min(cv.ct$cptable[,"xerror"]),"CP"])
length(pruned.ct$frame$var[pruned.ct$frame$var == "<leaf>"])
prp(pruned.ct, type = 1, extra = 1, split.font = 1, varlen = -10,
    box.col=ifelse(pruned.ct$frame$var == "<leaf>", 'gray', 'white')) 
# this one looks better, more interpretable, balanced
printcp(pruned.ct)



##Ensemble methods

# random forest
library(randomForest)

# grid search
best_F1<-0
best_ntree<-0
best_mtry<-0
best_nodezise<-0
for (ntree in seq(from=98, to=101, by=1)) {
  for (mtry in seq(from=2, to=6,by=1)){
    for(nodesize in seq(from=3, to=8, by=1)){
      rf <- randomForest(as.factor(TARGET_deathRate) ~ ., data = train.df, ntree = ntree, 
                         mtry = mtry, nodesize = nodesize, importance = TRUE)     
      rf.pred <- predict(rf, valid.df)
      results_rf<-confusionMatrix(rf.pred, as.factor(valid.df$TARGET_deathRate))
      Sensitivity <- results_rf$byClass[1] # where did this come from?
      Specificity <- results_rf$byClass[2] 
      F1 <- (2 * Sensitivity * Specificity) / (Sensitivity + Specificity)
      if (F1 > best_F1) {
        best_F1 <- F1
        best_ntree <- ntree
        best_mtry <- mtry
        best_nodezise <- nodesize
      }}}}
cat("random forest F1=" , best_F1, "; best_ntree=", best_ntree, 
    "; best_mtry=", best_mtry,";best_nodezise=",best_nodezise)

# variable importance plot
varImpPlot(rf, type = 1)




#boosting
library(adabag)
library(rpart) 
library(caret)

train.df$TARGET_deathRate <- as.factor(train.df$TARGET_deathRate)
set.seed(2)

# grid search
best_F1<-0
best_mfinal<-0
best_maxdepth<-0

for (m_final in seq(from=10, to=15, by=1)) {
  for (max_depth in seq(from=5, to=7,by=1)) {
    boost <- boosting(TARGET_deathRate ~ .,data = train.df, mfinal=m_final,maxdepth=max_depth)
    boost_pred <- predict(boost, valid.df, type = "class")
    results_boost <- confusionMatrix(as.factor(boost_pred$class), as.factor(valid.df$TARGET_deathRate))
    Sensitivity <- results_boost$byClass[1] 
    Specificity <- results_boost$byClass[2]
    F1 <- (2 * Sensitivity * Specificity) / (Sensitivity + Specificity)
    if (F1 > best_F1) {
      best_F1 <- F1
      best_mfinal <- m_final
      best_maxdepth <- max_depth
    }}}
cat("boosting F1=" , best_F1, "; best_mfinal=", best_mfinal, 
    "; best_maxdepth=", best_maxdepth)
results_boost


#bagging
# grid search
best_F1<-0
best_nbagg <-0

for (nbagg in seq(from=14, to=18, by=1)) {
    bag <- bagging(TARGET_deathRate ~ ., data = train.df,nbagg = nbagg)
    bag_pred <- predict(bag, valid.df, type = "class")
    results_bagged<-confusionMatrix(as.factor(bag_pred$class), as.factor(valid.df$TARGET_deathRate))
    accuracy <- cm$overall[1]
    Sensitivity <- results$byClass[1] 
    Specificity <- results$byClass[2]
    F1 <- (2 * Sensitivity * Specificity) / (Sensitivity + Specificity)
    if (F1 > best_F1) {
      best_F1 <- F1
      best_nbagg <- nbagg
    }}
cat("bagging trees accuracy=", accuracy, " and ", " F1=", best_F1)
results_bagged
