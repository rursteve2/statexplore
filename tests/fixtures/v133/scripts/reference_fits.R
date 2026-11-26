# Independent R oracle, not application code. Run from package root after generate_fixtures.R.
# Required local packages: lme4, nlme. No network or auto install.
for (p in c("lme4","nlme")) if (!requireNamespace(p,quietly=TRUE)) stop("Missing local R package: ",p)
options(digits=17, contrasts=c("contr.treatment","contr.poly"))
out <- "reference/output"; dir.create(out,recursive=TRUE,showWarnings=FALSE)
metrics <- list(); statuses <- list(); components <- list(); fits <- new.env(parent=emptyenv())
add <- function(case,metric,value) { if(length(value)==1 && is.numeric(value) && !is.na(value)) metrics[[length(metrics)+1L]] <<- data.frame(case=case,metric=metric,value=as.numeric(value)) }
status <- function(case,state,warning="",error="") statuses[[length(statuses)+1L]] <<- data.frame(case=case,status=state,warnings=warning,error=error)
prepare <- function(d) {
  d <- d[complete.cases(d[c("SCHOOL","ID","TIME","GROUP","Y")]),]
  d$SCHOOL <- factor(d$SCHOOL); d$GROUP <- factor(d$GROUP,levels=c(0,1)); d$PID <- interaction(d$SCHOOL,d$ID,drop=TRUE,lex.order=TRUE)
  d$OCC <- match(d$TIME,sort(unique(d$TIME)))-1L; d$TIMEF <- factor(d$TIME)
  d <- d[order(d$PID,d$TIME),]; d
}
read_fixture <- function(n) prepare(read.csv(file.path(out,paste0(n,".csv"))))
d <- prepare(read.csv("fixtures/Test43B.csv"))
# Explicit model formulas independent of the StatExplore generator.
fit <- function(data, form, re, residual, clustered, method) {
  if(residual=="independent") {
    random <- switch(re,RI="(1|PID)",DIAG="(1+TIME||PID)",RS="(1+TIME|PID)")
    if(clustered) random <- paste("(1|SCHOOL)",random,sep="+")
    f <- as.formula(paste(paste(deparse(form),collapse=" "),random,sep="+"))
    lme4::lmer(f,data=data,REML=method=="REML",na.action=na.fail)
  } else {
    if(clustered) {
      random <- switch(re,RI=~1|SCHOOL/PID,DIAG=list(SCHOOL=nlme::pdDiag(~1),PID=nlme::pdDiag(~1+TIME)),RS=list(SCHOOL=nlme::pdDiag(~1),PID=nlme::pdSymm(~1+TIME)))
      corr <- if(residual=="AR1") nlme::corAR1(form=~OCC|SCHOOL/PID) else nlme::corCAR1(form=~TIME|SCHOOL/PID)
    } else {
      random <- switch(re,RI=~1|PID,DIAG=list(PID=nlme::pdDiag(~1+TIME)),RS=~1+TIME|PID)
      corr <- if(residual=="AR1") nlme::corAR1(form=~OCC|PID) else nlme::corCAR1(form=~TIME|PID)
    }
    nlme::lme(form,data=data,random=random,correlation=corr,method=method,na.action=na.fail,control=nlme::lmeControl(maxIter=200,msMaxIter=200))
  }
}
record <- function(case,m,data,form) {
  ll <- logLik(m); add(case,"n",nobs(m)); add(case,"participants",nlevels(data$PID)); add(case,"k",attr(ll,"df")); add(case,"logLik",ll); add(case,"AIC",AIC(m)); add(case,"BIC",BIC(m)); add(case,"BIC_n",if(inherits(m,"lme") && m$method=="REML") nobs(m)-length(nlme::fixef(m)) else nobs(m))
  add(case,"observed_clusters",nlevels(data$SCHOOL)); add(case,"modeled_clusters",if(inherits(m,"merMod")) as.integer("SCHOOL" %in% names(lme4::getME(m,"flist")))*nlevels(data$SCHOOL) else if("SCHOOL" %in% names(m$groups)) nlevels(data$SCHOOL) else 0)
  add(case,"residual_variance",sigma(m)^2)
  if(inherits(m,"merMod")) {
    b <- lme4::fixef(m); vv <- as.data.frame(lme4::VarCorr(m));
    X <- lme4::getME(m,"X"); Z <- lme4::getME(m,"Z"); L <- lme4::getME(m,"Lambda"); rv <- mean(Matrix::rowSums((Z %*% L)^2))*sigma(m)^2
    add(case,"singular",as.integer(lme4::isSingular(m))); cf <- summary(m)$coefficients
    for(j in seq_len(nrow(vv))) if(vv$grp[j]!="Residual") add(case,paste("cov",vv$grp[j],vv$var1[j],ifelse(is.na(vv$var2[j]),"variance",vv$var2[j]),sep=":"),vv$vcov[j])
  } else {
    b <- nlme::fixef(m); X <- model.matrix(form,data=data); X <- X[,names(b),drop=FALSE]; rv <- 0; cf <- summary(m)$tTable
    for(g in names(m$modelStruct$reStruct)) {
      G <- as.matrix(nlme::pdMatrix(m$modelStruct$reStruct[[g]]))*sigma(m)^2
      z <- cbind("(Intercept)"=1,TIME=data$TIME)[,colnames(G),drop=FALSE]
      rv <- rv+mean(rowSums((z%*%G)*z))
      for(a in seq_len(nrow(G))) for(h in seq_len(a)) add(case,paste("cov",g,rownames(G)[h],if(a==h) "variance" else rownames(G)[a],sep=":"),G[a,h])
    }
    if(!is.null(m$modelStruct$corStruct)) add(case,"phi",coef(m$modelStruct$corStruct,unconstrained=FALSE)[1])
  }
  for(nm in names(b)) { add(case,paste0("beta:",nm),b[[nm]]); add(case,paste0("se:",nm),cf[nm,2]); if(ncol(cf)>=5L) add(case,paste0("wald_p:",nm),cf[nm,5]) }
  fv <- var(as.numeric(X%*%b)); tv <- fv+rv+sigma(m)^2
  add(case,"R2_marginal",fv/tv); add(case,"R2_conditional",(fv+rv)/tv)
}
run <- function(case,data,form,re,residual,clustered,method) {
  warns <- character(); msgs <- character()
  m <- tryCatch(withCallingHandlers(fit(data,form,re,residual,clustered,method),warning=function(w){warns<<-c(warns,conditionMessage(w));invokeRestart("muffleWarning")},message=function(w){msgs<<-c(msgs,conditionMessage(w));invokeRestart("muffleMessage")}),error=function(e)e)
  if(inherits(m,"error")){status(case,"failed",paste(unique(c(warns,msgs)),collapse=" | "),conditionMessage(m));return(NULL)}
  fits[[case]] <- m
  e <- tryCatch({record(case,m,data,form);NULL},error=function(e)e)
  diag <- if(inherits(m,"merMod")) c(m@optinfo$conv$lme4$messages,if(lme4::isSingular(m))"singular" else NULL) else if(is.character(m$apVar)) paste("apVar:",m$apVar) else NULL
  status(case,if(inherits(e,"error")) "report_failed" else if(length(c(warns,msgs,diag))) "flagged" else "fitted",paste(unique(c(warns,msgs,diag)),collapse=" | "),if(inherits(e,"error")) conditionMessage(e) else "")
  m
}
lrt <- function(case,small,large,reference="chi") {
  if(is.null(small)||is.null(large)) {status(case,"unavailable",error="Fit failed"); return(invisible(NULL))}
  raw <- 2*(as.numeric(logLik(large))-as.numeric(logLik(small))); df <- attr(logLik(large),"df")-attr(logLik(small),"df"); add(case,"LR_raw",raw); add(case,"df",df)
  if(raw < -1e-6 || df<=0) {status(case,"withheld",error="Invalid negative LR or df");return(invisible(NULL))}
  lr <- max(0,raw); add(case,"LR",lr)
  p <- switch(reference,chi=pchisq(lr,df,lower.tail=FALSE),mix01=if(lr==0)1 else .5*pchisq(lr,1,lower.tail=FALSE),mix12=.5*pchisq(lr,1,lower.tail=FALSE)+.5*pchisq(lr,2,lower.tail=FALSE))
  add(case,"p",p); if(p>0)add(case,"log10_p",log10(p)); status(case,"computed",warning="Analytic reference only; application must separately qualify fit diagnostics and boundary assumptions")
}
full <- Y~TIME*GROUP; reduced <- Y~TIME+GROUP
# 18 x both workflows, A full and reduced, B fixed full part.
for(cl in c(FALSE,TRUE)) for(re in c("RI","DIAG","RS")) for(res in c("independent","AR1","CAR1")) {
  key <- paste(if(cl)"cluster" else "participant",re,res,sep="_")
  a <- run(paste0("A_",key,"_full"),d,full,re,res,cl,"ML")
  b <- run(paste0("A_",key,"_reduced"),d,reduced,re,res,cl,"ML")
  lrt(paste0("A_",key,"_focal"),b,a)
  run(paste0("B_",key),d,full,re,res,cl,"REML")
}
# 48A comparisons on regular data.
ri <- fits[["B_cluster_RI_independent"]]; diag <- fits[["B_cluster_DIAG_independent"]]; rs <- fits[["B_cluster_RS_independent"]]
lrt("48A_RI_DIAG",ri,diag,"mix01"); lrt("48A_RI_RS",ri,rs,"mix12"); lrt("48A_DIAG_RS",diag,rs)
# 45 reused IDs, exact equivalent grouping.
r <- read_fixture("Test45A_ReusedIDs")
run("45A_cluster_RS_AR1",r,full,"RS","AR1",TRUE,"ML")
tryCatch({m45 <- nlme::lme(full,data=r,random=~1|PID,method="REML",na.action=na.fail); record("45E_participant_RI",m45,r,full);status("45E_participant_RI","fitted")},error=function(e)status("45E_participant_RI","failed",error=conditionMessage(e)))
# 46 irregular/missing and 48B residual models (RI).
i <- read_fixture("Test46A_IrregularTime"); m <- read_fixture("Test46B_MissingTime")
a <- run("46A_AR1",i,full,"RI","AR1",TRUE,"REML"); c <- run("46A_CAR1",i,full,"RI","CAR1",TRUE,"REML")
# use nlme for independent residual to match 48B native REML BIC convention
tryCatch({base <- nlme::lme(full,data=i,random=~1|SCHOOL/PID,method="REML",na.action=na.fail);record("48B_nlme_independent",base,i,full);status("48B_nlme_independent","fitted");lrt("48B_ind_AR1",base,a);lrt("48B_ind_CAR1",base,c,"mix01")},error=function(e)status("48B_nlme_independent","failed",error=conditionMessage(e)))
run("46B_AR1",m,full,"RI","AR1",TRUE,"REML")
# 48C corrected time comparison preserves AR1; separate independent baseline.
for(res in c("AR1","independent")) {
  lin <- run(paste0("48C_",res,"_linear"),i,full,"RI",res,TRUE,"ML")
  quad <- run(paste0("48C_",res,"_quadratic"),i,Y~(TIME+I(TIME^2))*GROUP,"RI",res,TRUE,"ML")
  catf <- run(paste0("48C_",res,"_categorical"),i,Y~TIMEF*GROUP,"RI",res,TRUE,"ML")
  lrt(paste0("48C_",res,"_lin_quad"),lin,quad);lrt(paste0("48C_",res,"_quad_cat"),quad,catf);lrt(paste0("48C_",res,"_lin_cat"),lin,catf)
}
s <- read_fixture("Test49A_Singularity")
si <- run("49A_DIAG",s,full,"DIAG","independent",TRUE,"REML"); sr <- run("49B_RI",s,full,"RI","independent",TRUE,"REML"); sc <- run("49B_RS",s,full,"RS","independent",TRUE,"REML")
lrt("49B_RI_DIAG",sr,si,"mix01");lrt("49B_RI_RS",sr,sc,"mix12") # covariance-only p withheld by app when variance zero
run("50C_filtered",read_fixture("Test50C_Filtered"),full,"RS","AR1",TRUE,"ML")
write.csv(do.call(rbind,metrics),file.path(out,"reference_metrics.csv"),row.names=FALSE)
write.csv(do.call(rbind,statuses),file.path(out,"reference_status.csv"),row.names=FALSE)
writeLines(capture.output(sessionInfo()),file.path(out,"R_sessionInfo.txt"))
# Fail on actual fit/report error; review expected flagged cases separately.
s <- do.call(rbind,statuses); if(any(s$status %in% c("failed","report_failed","unavailable"))) stop("Reference fit/report errors; inspect reference_status.csv")
cat("Reference calculations complete; flagged fits require explicit review.\n")
