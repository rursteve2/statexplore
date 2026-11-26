# Supplement: synthetic binary and ordinal longitudinal fixtures and independent mean-model comparisons.
# Execute with actual local R dependencies; no package downloads.
for(p in c("lme4","ordinal","geepack")) if(!requireNamespace(p,quietly=TRUE)) stop("Missing local package: ",p)
options(digits=17, contrasts=c("contr.treatment","contr.poly")); set.seed(13302)
out <- "reference/output"; dir.create(out,recursive=TRUE,showWarnings=FALSE)
d <- read.csv("fixtures/Test43B.csv"); d$PID <- as.integer(factor(d$ID)); d$SCHOOL <- factor(d$SCHOOL); d$GROUP <- factor(d$GROUP); d <- d[order(d$PID,d$TIME),]
eta <- -.5+.12*d$TIME+.25*(d$GROUP=="1")+.25*d$TIME*(d$GROUP=="1")+rnorm(160,0,.7)[d$PID]
d$BIN <- rbinom(nrow(d),1,plogis(eta)); d$ORD <- ordered(cut(eta+rlogis(nrow(d)),c(-Inf,-1,0,1,Inf),labels=FALSE))
write.csv(d,file.path(out,"NonGaussian.csv"),row.names=FALSE)
# Same rows/link/integration approximation. Preserve warnings in execution log.
b0 <- lme4::glmer(BIN~TIME+GROUP+(1|PID),data=d,family=binomial(),nAGQ=1)
b1 <- lme4::glmer(BIN~TIME*GROUP+(1|PID),data=d,family=binomial(),nAGQ=1)
o0 <- ordinal::clmm(ORD~TIME+GROUP+(1|PID),data=d,link="logit",nAGQ=1,Hess=TRUE)
o1 <- ordinal::clmm(ORD~TIME*GROUP+(1|PID),data=d,link="logit",nAGQ=1,Hess=TRUE)
fit_row <- function(case,m){ll<-logLik(m);data.frame(case, n=nobs(m),k=attr(ll,"df"),logLik=as.numeric(ll),AIC=AIC(m),BIC=BIC(m))}
write.csv(rbind(fit_row("GLMM_reduced",b0),fit_row("GLMM_full",b1),fit_row("CLMM_reduced",o0),fit_row("CLMM_full",o1)),file.path(out,"non_gaussian_likelihood.csv"),row.names=FALSE)
print(anova(b0,b1));print(anova(o0,o1)); print(summary(b1)); print(summary(o1))
# GEE no ordinary likelihood. Same mean, sample, working correlation alternatives.
g <- lapply(c("independence","exchangeable","ar1"),function(cor)geepack::geeglm(BIN~TIME*GROUP,data=d,id=PID,waves=TIME+1L,family=binomial(),corstr=cor))
names(g)<-c("independence","exchangeable","ar1")
for(n in names(g)) {cat("\n",n,"\n"); print(summary(g[[n]])); print(geepack::QIC(g[[n]])); b<-coef(g[[n]])[["TIME:GROUP1"]]; v<-vcov(g[[n]])["TIME:GROUP1","TIME:GROUP1"]; print(c(robust_Wald=b^2/v,p=pchisq(b^2/v,1,lower.tail=FALSE)))}
# Developer extends to actual random/time comparisons, support matrix, failures, and clustered variants where supported.
