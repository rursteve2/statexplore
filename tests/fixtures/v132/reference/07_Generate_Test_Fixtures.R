# Independent, privacy-safe fixture generator. Run from package root:
# Rscript --vanilla 07_Generate_Test_Fixtures.R <new output directory>
# No StatExplore functions, external data, or downloads. Original HRS not included.
args <- commandArgs(trailingOnly=TRUE)
root <- if(length(args)) args[1] else 'reference_run'
dir.create(root, recursive=TRUE, showWarnings=FALSE)
out <- file.path(root,'fixtures'); dir.create(out,showWarnings=FALSE)
if(length(list.files(out))) stop('Fixture directory is not empty; use a new run directory.')
RNGkind('Mersenne-Twister','Inversion','Rejection')
options(digits=17)
manifest <- list()
write_fixture <- function(id,d,seed=NA_integer_,basis='reconstructed') {
  path <- file.path(out,paste0(id,'.csv'))
  write.csv(d,path,row.names=FALSE,na='')
  # Fit references on serialized/reimported data, not on in-memory originals.
  z <- read.csv(path,check.names=FALSE)
  stopifnot(nrow(z)==nrow(d),ncol(z)==ncol(d))
  manifest[[id]] <<- data.frame(fixture=id,file=basename(path),seed=seed,basis=basis,
    rows=nrow(z),columns=ncol(z),md5=unname(tools::md5sum(path)))
  invisible(z)
}
panel <- function(n,nt=8) data.frame(ID=rep(seq_len(n),each=nt),TIME=rep(0:(nt-1),n))
noise_ar <- function(n,nt,phi) unlist(lapply(seq_len(n),function(i)
  as.numeric(stats::arima.sim(model=list(ar=phi),n=nt,sd=1))),use.names=FALSE)
add_group <- function(d) {d$GROUP <- as.integer(d$ID %% 2==0);d}
# 35C equivalent reconstruction: same coefficients/fit metrics, different row identity.
# x=1:10 repeated 4 times; error orthogonal to intercept and x, SSE=200.
x <- rep(1:10,each=4); e <- rep(c(-1,1,-1,1)*sqrt(5),10)
d <- data.frame(ROW_ID=1:80,FILTER=rep(c('A','B'),each=40),X=rep(x,2),
                Y=c(50+2*x+e,72-2*x+e))
write_fixture('T35C_Identity',d,basis='equivalent reconstruction; original row values unavailable')
# 35A equivalent ordinal reversal control; generate without separation intentionally.
set.seed(3501); n <- 500; x <- rnorm(n); latent <- .45*x+rlogis(n)
d <- data.frame(ROW_ID=seq_len(n),X=x,Y=as.integer(cut(latent,c(-Inf,-1.4,-.5,.5,1.4,Inf))))
write_fixture('T35A_Ordinal',d,3501,'new equivalent control; not HRS historical fixture')
# 36A/B: documented seed and generation sequence. Revalidate historical metrics; exact original bytes absent.
for(k in c('36A','36B')) {
  n <- if(k=='36A') 100 else 60; seed <- if(k=='36A') 3601 else 3602
  phi <- if(k=='36A') -.55 else .20; set.seed(seed); d <- panel(n)
  b <- rnorm(n,0,2); e <- noise_ar(n,8,phi)
  d$Y <- 20+.5*d$TIME+b[d$ID]+e
  write_fixture(paste0('T',k),d,seed)
}
set.seed(3603); d <- panel(80); z1 <- rnorm(80); z2 <- rnorm(80)
b <- 2*z1; s <- .35*(.30*z1+sqrt(1-.30^2)*z2); e <- noise_ar(80,8,.25)
d$Y <- 20+.5*d$TIME+b[d$ID]+s[d$ID]*d$TIME+e
write_fixture('T36C',d,3603); write_fixture('T36D',add_group(d),3603)
set.seed(3701); d <- panel(80); b <- rnorm(80,0,2)
d$Y <- 20+.5*d$TIME+b[d$ID]+rnorm(nrow(d))
write_fixture('T37A',d,3701); write_fixture('T37A_Discovery',add_group(d),3701)
extreme <- function(seed) {
  set.seed(seed); d <- panel(80); z <- rnorm(80)
  d$Y <- 20+.5*d$TIME+2*z[d$ID]-.45*z[d$ID]*d$TIME+rnorm(nrow(d));d
}
write_fixture('T37B',extreme(3704),3704)
write_fixture('T37D_3703',add_group(extreme(3703)),3703,'reconstructed; warning not stable after CSV')
write_fixture('T37D_3702',add_group(extreme(3702)),3702,'reconstructed CSV-stable warning in historical R environment')
# 38: verbatim documented generation order and formula.
set.seed(3801); ns<-20; np<-8; nt<-6; ni<-ns*np
d <- data.frame(SCHOOL=rep(1:ns,each=np*nt),ID=rep(1:ni,each=nt),TIME=rep(0:(nt-1),ni))
d$GROUP <- ifelse(d$ID%%2==0,1,0)
school_intercept <- rnorm(ns,0,2.5); z1<-rnorm(ni);z2<-rnorm(ni)
participant_intercept <- 1.5*z1
participant_slope <- .35*(.30*z1+sqrt(1-.30^2)*z2)
e <- noise_ar(ni,nt,.35)
d$Y <- 20+.5*d$TIME+.8*d$GROUP+school_intercept[d$SCHOOL]+participant_intercept[d$ID]+participant_slope[d$ID]*d$TIME+e
stopifnot(nrow(d)==960,length(unique(d$ID))==160,length(unique(d$SCHOOL))==20,!anyDuplicated(d[c('SCHOOL','ID','TIME')]))
write_fixture('T38',d,3801,'documented exact generator; bytes revalidated locally')
d2 <- d; d2$ID <- ((d2$ID-1)%%np)+1L
write_fixture('CD_ReusedID',d2,3801,'new nested local-ID control')
d3<-d; d3$TIME<-c(0,1,2,4,7,11)[d3$TIME+1];d3<-d3[!(d3$ID%%7==0 & d3$TIME==2),]
write_fixture('CD_Irregular',d3,3801,'new irregular/gap control')
d4<-d; d4$GROUP[d4$ID%%9==0 & d4$TIME==0]<-NA
write_fixture('CD_Missing',d4,3801,'new predictor-missing control')
d5<-rbind(d,d[1,]);write_fixture('CD_DuplicateTime',d5,3801,'new invalid duplicate control')
d6<-d;d6$SCHOOL[1]<-2L;write_fixture('CD_Crossing',d6,3801,'new invalid crossing under global-ID policy')
# Synthetic non-Gaussian and incomplete longitudinal control for Tests 28/29.
set.seed(2801);d<-add_group(panel(120,6));b<-rnorm(120,0,.9)
eta<--.8+.3*d$GROUP-.12*d$TIME+b[d$ID]
d$BINARY<-rbinom(nrow(d),1,plogis(eta));d$ORDINAL<-as.integer(cut(eta+rlogis(nrow(d)),c(-Inf,-1,0,1,Inf)))
d$Y<-20+.4*d$TIME+.7*d$GROUP+b[d$ID]+rnorm(nrow(d))
d$X<-rnorm(nrow(d));d$X[d$TIME==0]<-NA
write_fixture('T28_29_Synthetic',d,2801,'new family and missingness control; not HRS')
# Local code/metadata fixture; SAV creation needs a separate pyreadstat/haven test.
d<-data.frame(ID=1:10,X=c(1:7,99,9,10),Y=c(2,4,5,8,9,11,14,16,NA,20),GROUP=c(rep(0,5),rep(1,4),99))
write_fixture('T33_MissingCodes',d,basis='new SPSS metadata control; user missing X=99, GROUP=99; complete n=7')
# Copy supplied non-sensitive exact CSVs into run directory, retaining bytes.
for(f in c('StatExplore_v1.3.0_Whitespace_Regression_Fixture.csv','StatExplore_v131_Test26C_Sparse.csv')) {
  src<-file.path('fixtures',f); if(!file.exists(src)) stop('Run from package root: missing ',src)
  dest<-file.path(out,f); if(!file.copy(src,dest)) stop('copy failed ',src)
  z<-read.csv(dest); manifest[[f]]<-data.frame(fixture=sub('\\.csv$','',f),file=f,seed=NA_integer_,basis='supplied exact bytes',rows=nrow(z),columns=ncol(z),md5=unname(tools::md5sum(dest)))
}
write.csv(do.call(rbind,manifest),file.path(root,'generated_fixture_manifest.csv'),row.names=FALSE)
writeLines(c(R.version.string,capture.output(RNGkind()),capture.output(sessionInfo())),file.path(root,'fixture_session.txt'))
cat('Fixture generation completed:',normalizePath(out),'\n')
