# Test 51H-3: independent school bootstrap + stale reduced-cache counterfactual.
# Run in RStudio with lme4 installed. Select the two exact input files prompted.
library(lme4)
options(digits = 15)
cat('Select StatExplore_Test51C_ThreeWayInteraction.csv\n')
d <- read.csv(file.choose())
cat('Select validation_code(8).R (the original bootstrap export)\n')
expr <- parse(file = file.choose())
# Evaluate only the literal data.frame assignments, not the app's fitting code.
keep <- vapply(as.list(expr), function(x) is.call(x) &&
  identical(x[[1]], as.name('<-')) && identical(x[[2]], as.name('sx_vdraws')) &&
  is.call(x[[3]]) && identical(x[[3]][[1]], as.name('data.frame')), logical(1))
stopifnot(sum(keep) == 2L)
env <- new.env()
draws <- do.call(rbind, lapply(as.list(expr)[keep], function(x) {
  eval(x, envir = env); env$sx_vdraws
}))
stopifnot(nrow(d) == 1600, all(c('TestScore','Sleep','Anxiety','Attendance','Time','Gender','School','StudentID') %in% names(d)),
          !anyNA(d[c('TestScore','Sleep','Anxiety','Attendance','Time','Gender','School','StudentID')]),
          length(unique(d$School)) == 20, length(unique(d$StudentID)) == 160,
          nrow(draws) == 32000, all(sort(unique(draws$b)) == 1:20), length(unique(draws$b)) == 20,
          all(draws$row_id %in% seq_len(nrow(d))))
d$Gender <- factor(d$Gender)
terms <- c('Sleep','Anxiety','Attendance','Sleep + Anxiety','Sleep * Anxiety',
 'Sleep + Attendance','Sleep * Attendance','Anxiety + Attendance','Anxiety * Attendance',
 'Sleep + Anxiety + Attendance','Sleep * Anxiety + Attendance','Sleep * Attendance + Anxiety',
 'Anxiety * Attendance + Sleep',
 'Sleep + Anxiety + Attendance + Sleep:Anxiety + Sleep:Attendance',
 'Sleep + Anxiety + Attendance + Sleep:Anxiety + Anxiety:Attendance',
 'Sleep + Anxiety + Attendance + Sleep:Attendance + Anxiety:Attendance',
 'Sleep + Anxiety + Attendance + Sleep:Anxiety + Sleep:Attendance + Anxiety:Attendance',
 'Sleep * Anxiety * Attendance')
# -1 denotes required-only baseline; remaining references are zero-based IDs.
reference <- c(-1,-1,-1,-1,3,-1,5,-1,7,-1,9,9,9,9,9,9,9,9)
app_id <- c(17,13,13,13,13,13,13,16,13,13,16,17,16,17,17,17,17,17,17,17)
app_beta <- c(-2.292,-2.811,-2.908,-2.576,-2.680,-2.666,-2.360,-2.683,-2.553,-2.804,-2.909,-2.441,-2.503,-2.462,-2.424,-2.446,-2.451,-2.437,-2.443,-2.333)
fit_model <- function(x, data) {
  # Same ML model structure as the successful holdout/CV reference.
  f <- as.formula(paste('TestScore ~', x, '+ Time + Gender + (1 | School) + (1 + Time | StudentID)'))
  lmer(f, data = data, REML = FALSE)
}
get_coef <- function(fit, name) {
  beta <- fixef(fit); target <- sort(strsplit(name, ':', fixed=TRUE)[[1]])
  idx <- which(vapply(strsplit(names(beta), ':', fixed=TRUE), function(x) identical(sort(x),target), logical(1)))
  if (length(idx) == 1L) unname(beta[idx]) else NA_real_
}
all_scores <- list(); summaries <- list(); chunk_ll <- NULL; chunk_k <- NULL
for (b in 1:20) {
  cat('\n--- BOOTSTRAP DRAW', b, '---\n')
  dr <- draws[draws$b == b, ]
  x <- d[dr$row_id, , drop=FALSE]
  # Each sampled school and student must reproduce its complete original history.
  school_ok <- vapply(split(dr$row_id, dr$draw_cluster), function(ids) {
    orig <- unique(d$School[ids]); length(orig)==1L && identical(sort(as.integer(ids)),which(d$School==orig))
  }, logical(1))
  student_ok <- vapply(split(dr$row_id, dr$draw_gid), function(ids) {
    orig <- unique(d$StudentID[ids]); length(orig)==1L && identical(sort(as.integer(ids)),which(d$StudentID==orig))
  }, logical(1))
  stopifnot(nrow(x)==1600, length(school_ok)==20, all(school_ok),
            length(student_ok)==160, all(student_ok),
            all(tapply(dr$draw_cluster, dr$draw_gid, function(z) length(unique(z)))==1))
  original_schools <- length(unique(x$School))
  x$School <- factor(dr$draw_cluster)
  x$StudentID <- factor(dr$draw_gid)
  x <- x[order(x$StudentID,x$Time), ,drop=FALSE]
  rownames(x) <- paste0('b',seq_len(nrow(x)))
  # Fit every model on THIS draw only, including the baseline.
  fits <- lapply(c('1',terms), fit_model, data=x)
  ll <- vapply(fits,function(m) as.numeric(logLik(m)),numeric(1))
  k <- vapply(fits,function(m) as.numeric(attr(logLik(m),'df')),numeric(1))
  r <- reference + 2L; f <- 2:19
  lrt <- 2*(ll[f]-ll[r]); df <- k[f]-k[r]
  stopifnot(all(df>0))
  p <- pchisq(pmax(0,lrt),df,lower.tail=FALSE)
  logp <- pchisq(pmax(0,lrt),df,lower.tail=FALSE,log.p=TRUE)/log(10)
  # Counterfactual: mimic the known defect: reduced LL from the first draw
  # of each exported 10-draw chunk; full LL always from current draw.
  if (b %in% c(1,11)) { chunk_ll <- ll; chunk_k <- k }
  stale_lrt <- 2*(ll[f]-chunk_ll[r]); stale_df <- k[f]-chunk_k[r]
  stale_p <- pchisq(pmax(0,stale_lrt),stale_df,lower.tail=FALSE)
  stale_logp <- pchisq(pmax(0,stale_lrt),stale_df,lower.tail=FALSE,log.p=TRUE)/log(10)
  # App uses which.min(raw p); log p also checks for underflow-induced ties.
  fresh_id <- which.min(p)-1L; log_id <- which.min(logp)-1L
  stale_id <- which.min(stale_p)-1L; stale_log_id <- which.min(stale_logp)-1L
  beta <- get_coef(fits[[app_id[b]+2L]],'Sleep:Anxiety')
  true_beta <- get_coef(fits[[19]],'Sleep:Anxiety:Attendance')
  all_scores[[b]] <- data.frame(Draw=b,Candidate=0:17,Terms=terms,Reference=reference,
    Full_LL=ll[f],Fresh_reduced_LL=ll[r],Chunk_reduced_LL=chunk_ll[r],df=df,
    Fresh_LRT=lrt,Fresh_p=p,Fresh_log10_p=logp,Stale_LRT=stale_lrt,Stale_p=stale_p,
    Stale_log10_p=stale_logp,AIC=vapply(fits[f],AIC,numeric(1)),BIC=vapply(fits[f],BIC,numeric(1)),
    Singular=vapply(fits[f],isSingular,logical(1)),
    Diagnostics=vapply(fits[f],function(m) paste(m@optinfo$conv$lme4$messages,collapse='; '),character(1)))
  summaries[[b]] <- data.frame(Draw=b,Original_schools=original_schools,Rows=nrow(x),
    Draw_schools=nlevels(x$School),Draw_students=nlevels(x$StudentID),App=app_id[b],
    Fresh_raw=fresh_id,Fresh_log=log_id,Stale_raw=stale_id,Stale_log=stale_log_id,
    Fresh_zero_p=sum(p==0),Stale_zero_p=sum(stale_p==0),App_beta_rounded=app_beta[b],
    R_beta_app_model=beta,Beta_difference=beta-app_beta[b],Three_way_beta=true_beta,
    Fresh_matches_app=fresh_id==app_id[b],Stale_matches_app=stale_id==app_id[b])
  print(summaries[[b]],row.names=FALSE)
}
summary <- do.call(rbind,summaries); scores <- do.call(rbind,all_scores)
cat('\n--- DRAW-BY-DRAW COMPARISON ---\n'); print(summary,row.names=FALSE)
cat('\n--- SELECTION FREQUENCIES ---\n')
freq <- sapply(summary[c('App','Fresh_raw','Fresh_log','Stale_raw','Stale_log')],function(v) tabulate(v+1L,nbins=18))
rownames(freq) <- 0:17
print(freq)
cat('\n--- MATCHES TO STATEXPLORE ---\n')
print(c(Fresh_raw=sum(summary$Fresh_matches_app),Fresh_log=sum(summary$Fresh_log==summary$App),
        Stale_raw=sum(summary$Stale_matches_app),Stale_log=sum(summary$Stale_log==summary$App)))
cat('\n--- MAXIMUM DIFFERENCES ---\n')
print(c(Max_LRT_change=max(abs(scores$Fresh_LRT-scores$Stale_LRT)),
        Max_rounded_beta_difference=max(abs(summary$Beta_difference))))
cat('\n--- VERSION ---\n'); print(R.version.string); print(packageVersion('lme4'))
write.csv(summary,'Test51H3_Bootstrap_Comparison.csv',row.names=FALSE)
write.csv(scores,'Test51H3_Bootstrap_AllCandidates.csv',row.names=FALSE)
cat('\nCSV files saved in: ',getwd(),'\n',sep='')
