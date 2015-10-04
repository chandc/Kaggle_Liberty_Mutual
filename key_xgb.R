# This script is for the Liberty Mutual Competition

library(ggplot2)
library(randomForest)
library(readr)
library(caTools)
library(rpart)
library(caret)
library(glmnet)
library(xgboost)
library(e1071)


source("./Normalized_Gini.R")
set.seed(2015)

cat("Reading data\n")
Main <- read_csv("train.csv")
Mutual<- read_csv("test.csv")

#High_Hazard <- subset(Main, Main$Hazard > 2 & Main$Hazard < 6)
#nrow(High_Hazard)
#head(High_Hazard)
#summary(High_Hazard$Hazard)
#hist(High_Hazard$Hazard)
#Main <- High_Hazard

M5 <- rbind(Main[,3:34],Mutual[,2:33])
MainDummy <- dummyVars("~ .",data=M5, fullRank=F)
x_all <- as.data.frame(predict(MainDummy,M5))

DF <- x_all[1:nrow(Main),]
DF$Hazard <- Main$Hazard
Submit <- x_all[(nrow(Main)+1):nrow(x_all),]
Submit$Id <- Mutual$Id


# Split the data
set.seed(888)
spl = sample.split(DF$Hazard, SplitRatio = 0.8)
Train    <- subset(DF, spl==TRUE)
Validate <- subset(DF, spl==FALSE)

model = cv.glmnet(
  x = as.matrix(Train[,-112]), 
  y = Train$Hazard, 
  standardize = T,
  family = "poisson",
  alpha = 0.5,
  type.measure = "auc",
  intercept = T,
  nfolds = 10)

cat("Max cv.glmnet VAL AUC:", max(model$cvm, na.rm = T), "\n")

# And then make predictions on the test set:
pred_glmnet = predict(model, as.matrix(Validate[,-112]),s = "lambda.min")
sqrt(mean((Validate$Hazard-pred_glmnet)^2))
#plot(Train$Hazard,pred_glmnet[,1],xlim=c(0,40),ylim=c(0,40))
NormalizedGini(as.vector(Validate$Hazard), as.vector(pred_glmnet))

# boosted tree model (gbm) adjust learning rate and and trees
gbmGrid <-  expand.grid(n.trees = c(150,200),
                        interaction.depth =  c(7,9),
                        shrinkage = c(0.1),
                        n.minobsinnode = c(20,40))
gbmControl <- trainControl(method='cv', number=3, returnResamp='none')

# run model

gbm2Model <- train(Train[,-112], Train$Hazard, 
                   method='gbm', 
                   trControl=gbmControl,  
                   preProc = c("center", "scale"),
                   tuneGrid = gbmGrid, verbose=F)

pred_gbm <- predict(object=gbm2Model, Validate[,-112])
plot(Validate$Hazard,pred_gbm,xlim=c(0,40),ylim=c(0,40),asp = 1)
sqrt(mean((Validate$Hazard-pred_gbm)^2))
NormalizedGini(as.vector(Validate$Hazard), as.vector(pred_gbm))

# display variable importance on a +/- scale 
vimp <- varImp(gbm2Model, scale=F)
results <- data.frame(row.names(vimp$importance),vimp$importance$Overall)
results$VariableName <- rownames(vimp)
colnames(results) <- c('VariableName','Weight')
results <- results[order(results$Weight),]
results <- results[(results$Weight != 0),]

par(mar=c(5,15,4,2)) # increase y-axis margin. 
xx <- barplot(results$Weight[1:30], width = 0.85, 
              main = paste("Variable Importance -","Hazard"), horiz = T, 
              xlab = "< (-) importance >  < neutral >  < importance (+) >", axes = FALSE, 
              col = ifelse((results$Weight > 0), 'blue', 'red')) 
axis(2, at=xx, labels=results$VariableName[1:30], tick=FALSE, las=2, line=-0.3, cex.axis=0.6)  




cat("Training model\n")
rf <- randomForest(Train[,-112], Train$Hazard, ntree=100, imp=TRUE, sampsize=10000, do.trace=TRUE)

cat("Making predictions\n")
pred_rf<- predict(rf, Validate[,-112])
rmse_rf <- sqrt(mean((Validate$Hazard-pred_rf)^2))
NormalizedGini(Validate$Hazard,pred_rf)

#
# xgboost
#
param <- list("objective" = "count:poisson",
              "eta" = 0.002,
              "min_child_weight" = 5,
              "subsample" = .8,
              "colsample_bytree" = .8,
              "scale_pos_weight" = 1.0,
              "max_depth" = 8)
num_rounds <- 3000

dtrain <- xgb.DMatrix(as.matrix(Train[,-112]), label=as.vector(Train[,112]))
dval   <- xgb.DMatrix(as.matrix(Validate[,-112]), label=as.vector(Validate[,112]))
watchlist <- list(val=dval, train=dtrain)

bst2b <- xgb.train(params = param, data=dtrain, nround=num_rounds, print.every.n = 50, 
                   watchlist=watchlist, early.stop.round = 50, maximize = FALSE)
preds2b <- predict(bst2b,xgb.DMatrix(as.matrix(Validate[,-112])))
sqrt(mean((Validate$Hazard-preds2b)^2))
NormalizedGini(Validate$Hazard,preds2b)
resid <- abs(Validate$Hazard-preds2b)
#plot(preds2b,resid,xlim=c(0,50),ylim=c(0,50),asp = 1)
#plot(Validate$Hazard,preds2b,xlim=c(0,50),ylim=c(0,50),asp = 1)

rank_1 <- rank(pred_glmnet[,1])/length(pred_rf)
rank_2 <- rank(pred_rf)/length(pred_rf)
rank_3 <- rank(preds2b)/length(pred_rf)

rank_val <- rank(Validate$Hazard)/length(Validate$Hazard)
#bb <- data.frame(hazard=rank_val,glm=rank_1,rf=rank_2,xgb=rank_3)
#bb <- data.frame(hazard=Validate$Hazard,glm=pred_glmnet[,1],rf=pred_rf,xgb=preds2b)
#bb <- data.frame(hazard=rank_val,glm=rank_1,xgb=rank_3)

bb <- data.frame(hazard=Validate$Hazard,glm=rank_1,xgb=rank_3)
hybrid_model <- glm(hazard~.,data=bb)
pred_rank <- predict(hybrid_model,bb[,-1])
NormalizedGini(Validate$Hazard,pred_rank)


glm_meta = cv.glmnet(
  x = as.matrix(bb[,-1]),y=Validate$Hazard,
  standardize = T,
#  family = "poisson",
  alpha = 0.5,
  intercept = T,
  nfolds = 10)

pred_glm_hybrid = predict(glm_meta, as.matrix(bb[,-1]),s = "lambda.1se")
NormalizedGini(Validate$Hazard,pred_glm_hybrid[,1])



svm_model <- svm(hazard~.,data=bb,type="eps-regression",
                 kernel="radial",cross=10,
#                 cost=100,
#                 gamma=1.0
             cost=20,
             gamma=0.5

                 )
pred_rank <- predict(svm_model,bb[,-1])
NormalizedGini(Validate$Hazard,pred_rank)
resid <- sqrt((fitted(svm_model) - Validate$Hazard)^2)
#plot(Validate$Hazard,resid,xlim=c(0,50),ylim=c(0,50),asp = 1)
#plot(Validate$Hazard,pred_rank,xlim=c(0,50),ylim=c(0,50),asp = 1)



pred_1 <- predict(model, as.matrix(Submit[,-112]),s="lambda.1se")
pred_2 <- predict(rf, Submit[,-112])
pred_3 <- predict(bst2b,xgb.DMatrix(as.matrix(Submit[,-112])))

#rank_1 <- rank(pred_1[,1])/length(pred_2)
#rank_2 <- rank(pred_2)/length(pred_2)
#rank_3 <- rank(pred_3)/length(pred_2)
#b5 <- data.frame(glm=rank_1,rf=rank_2,xgb=rank_3)
#b5 <- data.frame(glm=pred_1[,1],rf=pred_2,xgb=pred_3)
b5 <- data.frame(glm=pred_1[,1],xgb=pred_3)

pred_sub <- predict(svm_model,b5)

pred_glm_hybrid_submit = predict(glm_meta, as.matrix(b5),s = "lambda.1se")

cat("Making Submission\n")
submission <- data.frame(Id=Submit$Id,Hazard= pred_sub)
write.table(submission, file = 'svm_meta_gamma_0.5.csv', row.names = F, col.names = T, sep = ",", quote = F)

submission <- data.frame(Id=Submit$Id,Hazard= pred_glm_hybrid_submit[,1])
write.table(submission, file = 'glmnet_meta.csv', row.names = F, col.names = T, sep = ",", quote = F)

