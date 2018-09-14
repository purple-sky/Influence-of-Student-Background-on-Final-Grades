library(MASS)
library(glmnet)
# loading data
d1=read.table("student-mat.csv",sep=",",header=TRUE)
d2=read.table("student-por.csv",sep=",",header=TRUE)

#d3=merge(d1,d2,by=c("school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet"))
d3 <- rbind(d1,d2)
d3.red <- d3[,-c(31,32)]
print(nrow(d3)) # 382 students
# setting up dataframes
#xm <- as.matrix(d3[,-c(31,32,33)])
xm <- data.matrix(d3[,-c(31,32,33)], rownames.force = NA)
y <- as.vector(as.numeric(d3$G3))
dat_X <- subset(d3, select=-c(31,32))

# full model on full data
full <- lm(G3~., data=dat_X)

# lambdas
lam <- exp( seq(-7, 7, length=42))
folds <- 5

fittedobject <- glmnet(x=xm, y=y, lambda=rev(lam), family='gaussian', alpha=0)
plot(fittedobject, xvar='lambda', label=TRUE, lwd=6, cex.axis=1.5, cex.lab=1.2, ylim=c(-2, 2))

# fitting RR with 10-fold CV
set.seed(12345)
model.rr <- cv.glmnet(x=xm, y=y, lambda=lam, nfolds=folds, alpha=0, family='gaussian')
plot(model.rr, lwd=6, cex.axis=1.5, cex.lab=1.2)
model.rr$lambda.min
#which.min(model.rr$cvm)
model.rr$glmnet.fit$beta[,which.min(model.rr$cvm)]

# singular LASSO
a <- glmnet(x=xm, y=y, lambda=rev(lam),family='gaussian', alpha=1, intercept=TRUE)
plot(a, xvar='lambda', label=TRUE, lwd=6, cex.axis=1.5, cex.lab=1.2)
# fitting LASSO 10-fold
set.seed(12345)
model.lasso <- cv.glmnet(x=xm, y=y, nfolds=folds, alpha=1, family='gaussian', intercept=TRUE)
plot(model.lasso, lwd=6, cex.axis=1.5, cex.lab=1.2)
log(model.lasso$lambda.min)
model.lasso$lambda.min
min(model.lasso$cvm)
coef(model.lasso, s=model.lasso$lambda.min)
#compare RR and LASSO
cbind(round(coef(model.rr, s='lambda.min'), 3),
      round(coef(model.lasso, s='lambda.min'), 3))

# Elastic net 10-fold CV
set.seed(12345)
model.en.75 <- cv.glmnet(x=xm, y=y, lambda=lam, nfolds=folds, alpha=0.75, 
                       family='gaussian', intercept=TRUE)
set.seed(12345)
model.en.05 <- cv.glmnet(x=xm, y=y, lambda=lam, nfolds=folds, alpha=0.05, 
                       family='gaussian', intercept=TRUE)
set.seed(12345)
model.en.1 <- cv.glmnet(x=xm, y=y, lambda=lam, nfolds=folds, alpha=0.1, 
                      family='gaussian', intercept=TRUE)
set.seed(12345)
model.en.5 <- cv.glmnet(x=xm, y=y, lambda=lam, nfolds=folds, alpha=0.5, 
                      family='gaussian', intercept=TRUE)
plot(model.en.75)
plot(model.en.05)
plot(model.en.1)
plot(model.en.5)

# MSPE comparisson between simple linear regression, RR, LASSO, and EN
#{r bigcompare, fig.width=5, fig.height=5, warning=FALSE, message=FALSE, echo=FALSE}
n <- nrow(xm)
k <- 5
ii <- (1:n) %% k + 1
set.seed(12345)
N <- 50
mspe.la <- mspe.st <- mspe.ri <- mspe.f <- rep(0, N)
for(i in 1:N) {
  ii <- sample(ii)
  pr.la <- pr.f <- pr.ri <- pr.st <- rep(0, n)
  for(j in 1:k) {
    tmp.ri <- cv.glmnet(x=xm[ii != j, ], y=y[ii != j], lambda=lam, 
                        nfolds=folds, alpha=0, family='gaussian') 
    tmp.la <- cv.glmnet(x=xm[ii != j, ], y=y[ii != j], lambda=lam, 
                        nfolds=folds, alpha=1, family='gaussian')
    null <- lm(G3 ~ 1, data=d3.red[ii != j, ])
    full <- lm(G3 ~ ., data=d3.red[ii != j, ])
    tmp.st <- stepAIC(null, scope=list(lower=null, upper=full), trace=FALSE)
    pr.ri[ ii == j ] <- predict(tmp.ri, s='lambda.min', newx=xm[ii==j,])
    pr.la[ ii == j ] <- predict(tmp.la, s='lambda.min', newx=xm[ii==j,])
    pr.st[ ii == j ] <- predict(tmp.st, newdata=d3.red[ii==j,])
    pr.f[ ii == j ] <- predict(full, newdata=d3.red[ii==j,])
  }
  mspe.ri[i] <- mean( (d3.red$G3 - pr.ri)^2 )
  mspe.la[i] <- mean( (d3.red$G3 - pr.la)^2 )
  mspe.st[i] <- mean( (d3.red$G3 - pr.st)^2 )
  mspe.f[i] <- mean( (d3.red$G3 - pr.f)^2 )
}
boxplot(mspe.la, mspe.ri, mspe.st, mspe.f, names=c('LASSO','Ridge', 'Stepwise', 'Full'), col=c('steelblue', 'gray80', 'tomato', 'springgreen'), cex.axis=1, cex.lab=1, cex.main=2)
mtext(expression(hat(MSPE)), side=2, line=2.5)

#{r bigcompare2, fig.width=5, fig.height=5, warning=FALSE, message=FALSE, echo=FALSE}
ii <- (1:n) %% k + 1
set.seed(12345)
N <- 50
mspe.en <- rep(0, N)
for(i in 1:N) {
  ii <- sample(ii)
  pr.en <- rep(0, n)
  for(j in 1:k) {
    tmp.en <- cv.glmnet(x=xm[ii != j, ], y=y[ii != j], lambda=lam, 
                        nfolds=folds, alpha=0.75, family='gaussian') 
    pr.en[ ii == j ] <- predict(tmp.en, s='lambda.min', newx=xm[ii==j,])
  }
  mspe.en[i] <- mean( (d3.red$G3 - pr.en)^2 )
}
boxplot(mspe.en, mspe.la, mspe.ri, mspe.st, mspe.f, 
        names=c('EN', 'LASSO','Ridge', 'Stepwise', 'Full'), 
        col=c('hotpink', 'steelblue', 'gray80', 'tomato', 'springgreen'), 
        cex.axis=1, cex.lab=1, cex.main=2)
mtext(expression(hat(MSPE)), side=2, line=2.5)
