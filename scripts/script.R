library(rpart)
library(randomForest)


# Load data, and merge on students that appear in both
# Math and Portuguese classes
fields = c("school","sex","age","address","famsize","Pstatus","Medu",
           "Fedu","Mjob","Fjob","reason","nursery","internet")
d.mat = read.table("student-mat.csv", sep=",", header=TRUE)
d.por = read.table("student-por.csv", sep=",", header=TRUE)
d.merge = merge(x = d.mat, y = d.por, by = fields)
print(nrow(d.merge))


# Pruning an overfitted tree, using CV to pick the best regression
# tree with the smallest xerror
N = 10
trees = vector('list', N)
xerr = numeric(N)
r.con = rpart.control(minsplit=3, cp=1e-8, xval=10)

set.seed(200)
for (i in 1:N) {
  tree = rpart(G3~school+sex+age+address+famsize+Pstatus+Medu+Fedu+Mjob+
                 Fjob+reason+guardian+traveltime+studytime+failures+
                 schoolsup+famsup+paid+activities+nursery+higher+internet+
                 romantic+famrel+freetime+goout+Dalc+Walc+health+absences,
               data=d.mat, method="class", control=r.con, parms=list(split='information'))
  
  cp.min.xe = tree$cptable[which.min(tree$cptable[,"xerror"]),"CP"]
  prune.tree = prune(tree, cp=cp.min.xe)
  trees[[i]] = prune.tree
  
  xerr[i] = min(tree$cptable[,"xerror"])
}

tree.mat = trees[[which.min(xerr)]]
print(tree.mat)
plot(tree.mat, uniform=FALSE, margin=0.01)
text(tree.mat, pretty=TRUE)

pred.tree = predict(tree.mat, newdata=d.por)
mspe.tree = mean(pred.tree - d.por$G3)


# Generate random forest with 1001 trees (error stabilizes after
# 500 trees)
set.seed(200)
rf.mat = randomForest(G3~school+sex+age+address+famsize+Pstatus+Medu+Fedu+Mjob+
                      Fjob+reason+guardian+traveltime+studytime+failures+
                      schoolsup+famsup+paid+activities+nursery+higher+internet+
                      romantic+famrel+freetime+goout+Dalc+Walc+health+absences,
                      data=d.mat, ntree=1001)
print(rf.mat)
plot(rf.mat)
varImpPlot(rf.mat)

pred.rf = predict(rf.mat, newdata=d.por)
mspe.rf = mean(pred.rf - d.por$G3)


# Bagging with 1001 trees
set.seed(200)
bag.mat = randomForest(G3~school+sex+age+address+famsize+Pstatus+Medu+Fedu+Mjob+
                       Fjob+reason+guardian+traveltime+studytime+failures+
                       schoolsup+famsup+paid+activities+nursery+higher+internet+
                       romantic+famrel+freetime+goout+Dalc+Walc+health+absences,
                       data=d.mat, mtry=30, ntree=1001)
print(bag.mat)
plot(bag.mat)
varImpPlot(bag.mat)

pred.bag = predict(bag.mat, newdata=d.por)
mspe.bag = mean(pred.bag - d.por$G3)


# Print and compare errors using trees
print(c(mspe.tree, mspe.rf, mspe.bag))
