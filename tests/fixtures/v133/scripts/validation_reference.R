# Independent Gaussian split/prediction and stability oracle (small illustrative RI candidate set).
# Extend to actual configured app selection rules/families; this is not the application harness.
if (!requireNamespace("lme4",quietly=TRUE)) stop("Missing local package lme4")
options(digits=17, contrasts=c("contr.treatment","contr.poly")); RNGkind("Mersenne-Twister","Inversion","Rejection")
out <- "reference/output"; dir.create(out,recursive=TRUE,showWarnings=FALSE)
d <- read.csv("fixtures/Test43B.csv"); d$ROW <- seq_len(nrow(d)); d$SCHOOL <- factor(d$SCHOOL); d$PID <- interaction(d$SCHOOL,d$ID,drop=TRUE); d$GROUP <- factor(d$GROUP,levels=c(0,1))
keys <- unique(d[c("SCHOOL","PID")]); k <- 5L
splits <- list(); scores <- list(); stability <- list(); preds <- list()
# One prespecified candidate selection rule: smallest training ML AIC among two models.
select_model <- function(train) {
  f0 <- lme4::lmer(Y~TIME+GROUP+(1|SCHOOL)+(1|PID),data=train,REML=FALSE)
  f1 <- lme4::lmer(Y~TIME*GROUP+(1|SCHOOL)+(1|PID),data=train,REML=FALSE)
  if(AIC(f1)<AIC(f0)) list(fit=f1,selected="interaction") else list(fit=f0,selected="additive")
}
predict_target <- function(f,test,target) {
  # Gaussian mean: new random effects zero. Existing school BLUP from training only.
  if(target=="school") predict(f,test,re.form=NA,allow.new.levels=TRUE) else predict(f,test,re.form=~(1|SCHOOL),allow.new.levels=TRUE)
}
for(target in c("school","participant")) for(mode in c("holdout","cv")) {
  set.seed(133)
  if(target=="school") {
    schools <- levels(d$SCHOOL); n <- length(schools)
    assignments <- if(mode=="holdout") sample(c(rep(1L,ceiling(.2*n)),rep(0L,n-ceiling(.2*n)))) else sample(rep(seq_len(k),length.out=n))
    fold <- assignments[match(d$SCHOOL,schools)]
  } else {
    lookup <- setNames(integer(nrow(keys)),as.character(keys$PID))
    for(s in levels(d$SCHOOL)) {
      ids <- as.character(keys$PID[keys$SCHOOL==s]); n <- length(ids)
      lookup[ids] <- if(mode=="holdout") sample(c(rep(1L,ceiling(.2*n)),rep(0L,n-ceiling(.2*n)))) else sample(rep(seq_len(k),length.out=n))
    }
    fold <- unname(lookup[as.character(d$PID)])
  }
  splits[[length(splits)+1L]] <- data.frame(target,mode,ROW=d$ROW,SCHOOL=d$SCHOOL,PID=d$PID,fold)
  for(f in if(mode=="holdout") 1L else seq_len(k)) {
    train <- d[fold!=f,]; test <- d[fold==f,]
    stopifnot(nrow(train)>0,nrow(test)>0,!any(test$PID %in% train$PID),!any(test$ROW %in% train$ROW))
    if(target=="school") stopifnot(!any(test$SCHOOL %in% train$SCHOOL)) else stopifnot(all(test$SCHOOL %in% train$SCHOOL))
    ans <- select_model(train); pred <- predict_target(ans$fit,test,target)
    changed <- test; changed$Y <- changed$Y+10000
    stopifnot(isTRUE(all.equal(pred,predict_target(ans$fit,changed,target))))
    error <- test$Y-pred; den <- sum((test$Y-mean(train$Y))^2)
    scores[[length(scores)+1L]] <- data.frame(target,mode,fold=f,train_n=nrow(train),test_n=nrow(test),selected=ans$selected,RMSE=sqrt(mean(error^2)),MAE=mean(abs(error)),R2=if(den>0)1-sum(error^2)/den else NA)
    preds[[length(preds)+1L]] <- data.frame(target,mode,fold=f,ROW=test$ROW,observed=test$Y,predicted=pred)
  }
}
# Small independent bootstrap example. Sampling/draw IDs explicit. Extend B for release testing.
B <- 20L
for(target in c("school","participant")) {
  set.seed(133)
  for(b in seq_len(B)) {
    pieces <- list(); j <- 0L
    schools <- levels(d$SCHOOL); draw <- if(target=="school") sample(schools,replace=TRUE) else schools
    for(i in seq_along(draw)) {
      sub <- d[d$SCHOOL==draw[i],]; ids <- unique(as.character(sub$PID)); ids <- if(target=="participant") sample(ids,replace=TRUE) else ids
      for(id in ids) {
        j <- j+1L; x <- sub[sub$PID==id,]; x$ORIGINAL_SCHOOL <- as.character(x$SCHOOL); x$ORIGINAL_PID <- as.character(x$PID)
        x$SCHOOL <- if(target=="school") paste0("draw_school_",i) else draw[i]; x$PID <- paste0("draw_pid_",j); pieces[[j]] <- x
      }
    }
    boot <- do.call(rbind,pieces); boot$SCHOOL <- factor(boot$SCHOOL); boot$PID <- factor(boot$PID)
    stopifnot(nlevels(boot$PID)==j,all(table(boot$PID)==6L),nlevels(boot$SCHOOL)==20L)
    ans <- tryCatch(select_model(boot),error=function(e)e)
    stability[[length(stability)+1L]] <- if(inherits(ans,"error")) data.frame(target,b,status="failed",selected=NA,interaction=NA,error=conditionMessage(ans)) else data.frame(target,b,status=if(lme4::isSingular(ans$fit))"flagged" else "fitted",selected=ans$selected,interaction=if(ans$selected=="interaction") lme4::fixef(ans$fit)[["TIME:GROUP1"]] else NA,error="")
  }
}
for(n in c("splits","scores","preds","stability")) write.csv(do.call(rbind,get(n)),file.path(out,paste0("validation_",n,".csv")),row.names=FALSE)
cat("Validation example complete; compare actual app splits/predictions for same rule, not arbitrary model selection.\n")
