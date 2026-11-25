# Independent reference fits; not the application regression runner.
# Run from package root after 07; requires local R packages, no downloads.
args <- commandArgs(trailingOnly=TRUE); root<-if(length(args)) args[1] else 'reference_run'
path <- file.path(root,'fixtures'); if(!dir.exists(path)) stop('Run generator 07 first.')
if(file.exists(file.path(root,'actual_metrics.csv'))) stop('Use a fresh run directory; reference evidence already exists.')
req<-c('lme4','lmerTest','nlme','performance','Matrix','ordinal','geepack','nnet')
missing<-req[!vapply(req,requireNamespace,logical(1),quietly=TRUE)]
if(length(missing)) stop('BLOCKED: missing local R packages: ',paste(missing,collapse=', '))
options(digits=17,contrasts=c('contr.treatment','contr.poly'))
metrics<-list(); diagnostics<-list(); attempts<-list(); fits<-new.env(parent=emptyenv()); errors<-0L
emit<-function(f,m,key,value) {
  if(length(value)!=1L) stop('Metric must be scalar: ',key)
  metrics[[length(metrics)+1L]]<<-data.frame(fixture=f,model=m,metric=key,value=as.numeric(value))
}
record<-function(f,m,key,value) diagnostics[[length(diagnostics)+1L]]<<-data.frame(fixture=f,model=m,diagnostic=key,value=paste(value,collapse='; '))
read_d<-function(f,local=FALSE) {
  d<-read.csv(file.path(path,paste0(f,'.csv')),check.names=FALSE)
  if('SCHOOL'%in%names(d)) d$SCHOOL<-factor(d$SCHOOL)
  if('ID'%in%names(d)) {
    d$ID<-factor(d$ID)
    if(local) d$ID<-interaction(d$SCHOOL,d$ID,drop=TRUE,lex.order=TRUE)
  }
  if('GROUP'%in%names(d)) d$GROUP<-factor(d$GROUP,levels=c(0,1))
  if(all(c('ID','TIME')%in%names(d))) {
    d<-d[if('SCHOOL'%in%names(d)) order(d$SCHOOL,d$ID,d$TIME) else order(d$ID,d$TIME),]
    d$OCC<-match(d$TIME,sort(unique(d$TIME)))-1L
  };d
}
# Captures warnings/errors without rewriting model specifications.
run_attempt<-function(f,m,optimizer,expr) {
  w<-character(); msg<-character()
  value<-tryCatch(withCallingHandlers(force(expr),warning=function(e){w<<-c(w,conditionMessage(e));invokeRestart('muffleWarning')},message=function(e){msg<<-c(msg,conditionMessage(e));invokeRestart('muffleMessage')}),error=function(e)e)
  bad<-inherits(value,'error')
  attempts[[length(attempts)+1L]]<<-data.frame(fixture=f,model=m,optimizer=optimizer,status=if(bad)'FAILED' else 'FITTED',warnings=paste(unique(w),collapse='; '),messages=paste(unique(msg),collapse='; '),error=if(bad)conditionMessage(value) else '')
  value
}
fit_lmer<-function(d,form,st,cluster=FALSE,reml=TRUE,alt=FALSE) {
  r<-switch(st,RI='(1|ID)',DIAG='(1+TIME||ID)',RS='(1+TIME|ID)')
  ff<-as.formula(paste(paste(deparse(form),collapse=''),'+',if(cluster)'(1|SCHOOL)+' else '',r))
  if(alt) {
    lmerTest::lmer(ff,data=d,REML=reml,na.action=na.fail,control=lme4::lmerControl(optimizer='bobyqa',optCtrl=list(maxfun=200000)))
  } else {
    lmerTest::lmer(ff,data=d,REML=reml,na.action=na.fail)
  }
}
fit_lme<-function(d,form,st,res,cluster=FALSE,reml=TRUE,opt='nlminb') {
  random<-if(cluster) switch(st,RI=~1|SCHOOL/ID,DIAG=list(SCHOOL=nlme::pdDiag(~1),ID=nlme::pdDiag(~1+TIME)),RS=list(SCHOOL=nlme::pdDiag(~1),ID=nlme::pdSymm(~1+TIME))) else switch(st,RI=~1|ID,DIAG=list(ID=nlme::pdDiag(~1+TIME)),RS=~1+TIME|ID)
  cf<-switch(res,Independent=NULL,AR1=nlme::corAR1(form=if(cluster) ~OCC|SCHOOL/ID else ~OCC|ID),CAR1=nlme::corCAR1(form=if(cluster) ~TIME|SCHOOL/ID else ~TIME|ID))
  nlme::lme(form,random=random,correlation=cf,data=d,method=if(reml)'REML' else 'ML',na.action=na.fail,control=nlme::lmeControl(opt=opt,msMaxIter=200))
}
collect<-function(f,m,a,d,cluster=FALSE) {
  if(inherits(a,'error')) {errors<<-errors+1L;record(f,m,'status','FAILED');record(f,m,'error',conditionMessage(a));return(invisible(NULL))}
  assign(paste(f,m,sep='::'),a,envir=fits);record(f,m,'status','FITTED');record(f,m,'formula',deparse(formula(a)))
  ll<-tryCatch(logLik(a),error=function(e)NULL)
  if(!is.null(ll)) {emit(f,m,'logLik',ll);emit(f,m,'parameters',attr(ll,'df'));emit(f,m,'AIC',AIC(a));emit(f,m,'BIC',BIC(a));bn<-attr(ll,'nobs');if(length(bn)==1L)emit(f,m,'BIC_n',bn)}
  emit(f,m,'n',nobs(a)); if('ID'%in%names(d))emit(f,m,'participants',length(unique(d$ID)))
  if('SCHOOL'%in%names(d))emit(f,m,'observed_clusters',length(unique(d$SCHOOL)))
  if(inherits(a,'merMod')) {
    fl<-lme4::getME(a,'flist');cn<-lme4::getME(a,'cnms');record(f,m,'groups',paste(names(fl),vapply(fl,nlevels,integer(1)),sep='='));record(f,m,'random_terms',paste(names(cn),vapply(cn,paste,character(1),collapse=','),sep='='))
    emit(f,m,'modeled_clusters',if('SCHOOL'%in%names(fl))nlevels(fl$SCHOOL) else 0L)
    if(nlevels(fl$ID)!=length(unique(d$ID)))stop('Reference participant mismatch: ',f,m)
    if(cluster != ('SCHOOL'%in%names(fl)))stop('Reference cluster mismatch: ',f,m)
    emit(f,m,'singular',lme4::isSingular(a));record(f,m,'optimizer',a@optinfo$optimizer);record(f,m,'optimizer_code',a@optinfo$conv$opt);record(f,m,'check_code',a@optinfo$conv$lme4$code);record(f,m,'convergence_messages',a@optinfo$conv$lme4$messages)
    v<-as.data.frame(lme4::VarCorr(a));
    for(i in seq_len(nrow(v))) {
      g<-v$grp[i];g<-if(g=='Residual')'Residual' else if(g=='SCHOOL')'SCHOOL' else 'ID'
      term<-if(is.na(v$var1[i]))'variance' else if(is.na(v$var2[i]))paste0('var_',v$var1[i]) else 'cov_intercept_TIME'
      emit(f,m,paste0(g,'.',term),v$vcov[i]); if(!is.na(v$var2[i]))emit(f,m,paste0(g,'.corr_intercept_TIME'),v$sdcor[i])
    }
    tab<-coef(summary(a)); rv<-mean(Matrix::rowSums((lme4::getME(a,'Z')%*%lme4::getME(a,'Lambda'))^2))*sigma(a)^2
    fv<-var(as.vector(lme4::getME(a,'X')%*%lme4::fixef(a)));ev<-sigma(a)^2
    if(inherits(a,'glmerMod')){rv<-fv<-ev<-NA_real_;record(f,m,'direct_r2','not applicable: Gaussian decomposition only')}
  } else if(inherits(a,'lme')) {
    gr<-a$groups;record(f,m,'groups',paste(names(gr),vapply(gr,function(x)length(unique(x)),integer(1)),sep='='))
    emit(f,m,'modeled_clusters',if('SCHOOL'%in%names(gr))length(unique(gr$SCHOOL)) else 0L)
    stopifnot(cluster==('SCHOOL'%in%names(gr)),length(unique(gr$ID))==length(unique(d$ID)))
    record(f,m,'apVar',if(is.matrix(a$apVar))'matrix available' else a$apVar)
    if(!is.null(a$modelStruct$corStruct)) {emit(f,m,'phi',coef(a$modelStruct$corStruct,unconstrained=FALSE)[1]);record(f,m,'correlation_formula',deparse(formula(a$modelStruct$corStruct)))}
    rv<-0;ev<-sigma(a)^2;fv<-var(as.numeric(predict(a,level=0)))
    for(g in names(a$modelStruct$reStruct)) {
      G<-nlme::pdMatrix(a$modelStruct$reStruct[[g]])*ev; en<-eigen(G,symmetric=TRUE,only.values=TRUE)$values
      record(f,m,paste0(g,'.covariance_terms'),colnames(G));emit(f,m,paste0(g,'.eigen_ratio'),min(en)/max(en))
      for(j in seq_len(nrow(G)))emit(f,m,paste0(g,'.var_',rownames(G)[j]),G[j,j])
      if(nrow(G)==2L){emit(f,m,paste0(g,'.cov_intercept_TIME'),G[1,2]);emit(f,m,paste0(g,'.corr_intercept_TIME'),cov2cor(G)[1,2])}
      Z<-model.matrix(if(nrow(G)==1L)~1 else ~TIME,d);rv<-rv+mean(rowSums((Z%*%G)*Z))
    }
    emit(f,m,'Residual.variance',ev);tab<-summary(a)$tTable
    ft<-anova(a,type='marginal');for(i in seq_len(nrow(ft)))emit(f,m,paste0('F_p.',rownames(ft)[i]),pf(ft[i,'F-value'],ft[i,'numDF'],ft[i,'denDF'],lower.tail=FALSE))
  } else {tab<-tryCatch(coef(summary(a)),error=function(e)NULL);rv<-fv<-ev<-NA_real_}
  if(is.matrix(tab))for(i in seq_len(nrow(tab)))for(j in seq_len(ncol(tab)))emit(f,m,paste0('coef.',rownames(tab)[i],'.',colnames(tab)[j]),tab[i,j])
  if(is.finite(rv)&&is.finite(fv)&&is.finite(ev)&&fv+rv+ev>0){emit(f,m,'R2_direct_m',fv/(fv+rv+ev));emit(f,m,'R2_direct_c',(fv+rv)/(fv+rv+ev))}
  if(inherits(a,c('merMod','lme'))) {
    rr<-tryCatch(withCallingHandlers(performance::r2_nakagawa(a),warning=function(w){record(f,m,'r2_warning',conditionMessage(w));invokeRestart('muffleWarning')}),error=function(e){record(f,m,'r2_error',conditionMessage(e));NULL})
    if(is.list(rr)){if(length(rr$R2_marginal)==1L)emit(f,m,'R2_package_m',rr$R2_marginal);if(length(rr$R2_conditional)==1L)emit(f,m,'R2_package_c',rr$R2_conditional)}
  }
  invisible(a)
}
run_mixed<-function(f,m,d,st,res='Independent',cluster=FALSE,form=Y~TIME,reml=TRUE,engine=if(res=='Independent')'lme4' else 'nlme',alt=FALSE) {
  if(engine=='lme4')a<-run_attempt(f,m,if(alt)'bobyqa' else 'default',fit_lmer(d,form,st,cluster,reml,alt)) else {
    a<-run_attempt(f,m,'nlminb',fit_lme(d,form,st,res,cluster,reml,'nlminb'))
    # Reproduce historical fallback only on recognizable optimizer failures.
    if(inherits(a,'error')&&grepl('convergence|iteration limit|evaluation limit|nlminb problem',conditionMessage(a),ignore.case=TRUE))a<-run_attempt(f,m,'optim',fit_lme(d,form,st,res,cluster,reml,'optim'))
  };collect(f,m,a,d,cluster)
}
# Raw LR and nominal probabilities. This is an independent numerical oracle, NOT permission
# to treat unresolved boundary/optimization cases as valid inference.
compare<-function(f,small,big,name,reference='ordinary') {
  a<-get0(paste(f,small,sep='::'),fits,inherits=FALSE);b<-get0(paste(f,big,sep='::'),fits,inherits=FALSE)
  if(is.null(a)||is.null(b)){record(f,name,'comparison','unavailable: failed model');return()}
  ll0<-logLik(a);ll1<-logLik(b);lr<-2*(as.numeric(ll1)-as.numeric(ll0));df<-attr(ll1,'df')-attr(ll0,'df');emit(f,name,'LR_raw',lr);emit(f,name,'df',df)
  if(!is.finite(lr)||lr< -1e-6||df<=0){record(f,name,'comparison','withhold nominal inference: inconsistent likelihood or df');return()}
  x<-max(0,lr);p<-pchisq(x,df,lower.tail=FALSE);emit(f,name,'p_ordinary',p)
  if(reference=='mix01')emit(f,name,'p_boundary',if(x==0)1 else .5*pchisq(x,1,lower.tail=FALSE))
  if(reference=='mix12')emit(f,name,'p_boundary',.5*pchisq(x,1,lower.tail=FALSE)+.5*pchisq(x,2,lower.tail=FALSE))
  record(f,name,'reference_distribution',reference)
}
# Identity counterexamples (not the original HRS ordinal data).
d<-read_d('T35C_Identity');for(g in c('A','B'))collect('T35C_Identity',g,lm(Y~X,data=d[d$FILTER==g,]),d[d$FILTER==g,])
d<-read_d('T35A_Ordinal');for(g in c('ascending','descending')){d$OY<-ordered(d$Y,levels=if(g=='ascending')1:5 else 5:1);collect('T35A_Ordinal',g,run_attempt('T35A_Ordinal',g,'clm',ordinal::clm(OY~X,data=d)),d)}
# Signed AR and positive CAR controls.
for(f in c('T36A','T36B')){
 d<-read_d(f);for(res in c('Independent','AR1','CAR1'))run_mixed(f,res,d,'RI',res,engine='nlme')
 compare(f,'Independent','AR1','RI_Independent_to_AR1');compare(f,'Independent','CAR1','RI_Independent_to_CAR1','mix01')
}
# Reference 36C/37C, both normal and boundary covariance controls.
for(f in c('T36C','T37A','T37B')){
 d<-read_d(f);for(st in c('RI','DIAG','RS')){
  run_mixed(f,paste0('lmer_',st),d,st);run_mixed(f,paste0(st,'_AR1'),d,st,'AR1')
 }
 if(f=='T36C')for(st in c('RI','DIAG','RS'))run_mixed(f,paste0(st,'_CAR1'),d,st,'CAR1')
 compare(f,'lmer_RI','lmer_RS','lmer_RI_to_RS','mix12');compare(f,'RI_AR1','DIAG_AR1','RI_to_DIAG_AR1','mix01')
}
d<-read_d('T36C');run_mixed('T36C','RS_Independent',d,'RS',engine='nlme');compare('T36C','RS_Independent','RS_CAR1','RS_Independent_to_CAR1','mix01')
d<-read_d('T36D');for(res in c('Independent','AR1','CAR1'))run_mixed('T36D',paste0('RI_',res),d,'RI',res,form=Y~GROUP+TIME,engine='nlme')
run_mixed('T36D','RI_CAR1_ML',d,'RI','CAR1',form=Y~GROUP+TIME,reml=FALSE);compare('T36D','RI_Independent','RI_CAR1','RI_Independent_to_CAR1','mix01')
# CSV-stable convergence and Discovery comparisons.
for(f in c('T37A_Discovery','T37D_3702','T37D_3703')){
 d<-read_d(f);for(st in c('RI','DIAG','RS'))run_mixed(f,st,d,st,form=Y~GROUP+TIME)
 run_mixed(f,'RS_TIME_first',d,'RS',form=Y~TIME+GROUP)
 if(f=='T37D_3702')run_mixed(f,'RS_bobyqa',d,'RS',form=Y~GROUP+TIME,alt=TRUE)
 compare(f,'RI','RS','RI_to_RS','mix12');compare(f,'DIAG','RS','DIAG_to_RS');compare(f,'RI','DIAG','RI_to_DIAG','mix01')
}
# Test 38: B (REML, fixed TIME+GROUP); A (ML, full/reduced); time-only B.
# Independent backend lme4, correlated residual backend nlme.
for(f in c('T38','CD_ReusedID','CD_Irregular','CD_Missing')){
 d<-read_d(f,local=f=='CD_ReusedID');d<-d[complete.cases(d[c('Y','TIME','GROUP','ID','SCHOOL')]),];d<-droplevels(d);d$OCC<-match(d$TIME,sort(unique(d$TIME)))-1L
 for(cl in c(FALSE,TRUE))for(st in c('RI','DIAG','RS'))for(res in c('Independent','AR1','CAR1')){
   m<-paste0(if(cl)'3L_' else '2L_',st,'_',res)
   run_mixed(f,m,d,st,res,cl,Y~TIME+GROUP)
   if(f=='T38'){
     run_mixed(f,paste0(m,'_A_full'),d,st,res,cl,Y~TIME+GROUP,reml=FALSE)
     run_mixed(f,paste0(m,'_A_reduced'),d,st,res,cl,Y~TIME,reml=FALSE)
     compare(f,paste0(m,'_A_reduced'),paste0(m,'_A_full'),paste0(m,'_A_GROUP'))
     if(cl)run_mixed(f,paste0(m,'_B_TIME_only'),d,st,res,cl,Y~TIME)
   }
 }
 if(f=='T38'){
  for(cl in c(FALSE,TRUE))for(st in c('RI','DIAG','RS'))run_mixed(f,paste0(if(cl)'3L_' else '2L_',st,'_nlme_Independent'),d,st,'Independent',cl,Y~TIME+GROUP,engine='nlme')
  for(res in c('Independent','AR1','CAR1')){compare(f,paste0('3L_RI_',res),paste0('3L_RS_',res),paste0('3L_RI_to_RS_',res),'mix12');compare(f,paste0('3L_RI_',res),paste0('3L_DIAG_',res),paste0('3L_RI_to_DIAG_',res),'mix01');compare(f,paste0('3L_DIAG_',res),paste0('3L_RS_',res),paste0('3L_DIAG_to_RS_',res));compare(f,paste0('2L_RS_',res),paste0('3L_RS_',res),paste0('Add_cluster_RS_',res),'mix01')}
  compare(f,'3L_RS_nlme_Independent','3L_RS_AR1','3L_RS_Independent_to_AR1');compare(f,'3L_RS_nlme_Independent','3L_RS_CAR1','3L_RS_Independent_to_CAR1','mix01')
 }
}
# Additional non-Gaussian reference fits (new synthetic control, not HRS numerics).
d<-read_d('T28_29_Synthetic');f<-'T28_29_Synthetic'
collect(f,'GLMM_RI',run_attempt(f,'GLMM_RI','glmer',lme4::glmer(BINARY~TIME+GROUP+(1|ID),d,family=binomial())),d)
collect(f,'CLMM_RI',run_attempt(f,'CLMM_RI','clmm',ordinal::clmm(ordered(ORDINAL)~TIME+GROUP+(1|ID),data=d)),d)
# GEE does not have likelihood; store QIC separately and do not fabricate logLik/AIC.
for(s in c('independence','exchangeable','ar1')){
 a<-run_attempt(f,paste0('GEE_',s),'geeglm',geepack::geeglm(Y~TIME+GROUP,data=d,id=ID,waves=OCC,corstr=s))
 m<-paste0('GEE_',s);if(inherits(a,'error')){errors<-errors+1L;record(f,m,'status','FAILED');next}
 record(f,m,'status','FITTED');emit(f,m,'n',nrow(d));emit(f,m,'participants',length(unique(d$ID)));q<-tryCatch(geepack::QIC(a),error=function(e){record(f,m,'QIC_error',conditionMessage(e));NULL});if(!is.null(q))for(k in names(q))emit(f,m,k,q[[k]])
}
# Independent sparse ordinal fixture: retain native test output, not invented expected values.
f<-'T26C';d<-read.csv(file.path(path,'StatExplore_v131_Test26C_Sparse.csv'));d$y<-ordered(d$y);d$g<-factor(d$g);d$z<-factor(d$z)
a<-run_attempt(f,'ordinal','clm',ordinal::clm(y~g*z,data=d));collect(f,'ordinal',a,d)
if(!inherits(a,'error'))record(f,'ordinal','nominal_test',tryCatch(capture.output(ordinal::nominal_test(a)),error=function(e)conditionMessage(e)))
write.csv(do.call(rbind,metrics),file.path(root,'actual_metrics.csv'),row.names=FALSE,na='')
write.csv(do.call(rbind,diagnostics),file.path(root,'reference_diagnostics.csv'),row.names=FALSE,na='')
write.csv(do.call(rbind,attempts),file.path(root,'reference_attempts.csv'),row.names=FALSE,na='')
writeLines(c(R.version.string,capture.output(sessionInfo()),paste(req,vapply(req,function(x)as.character(packageVersion(x)),character(1)),sep='=')),file.path(root,'reference_session.txt'))
cat('Reference fits completed; failed fits:',errors,'\n');if(errors>0L)quit(status=2L)
