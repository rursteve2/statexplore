# Run from package root: Rscript reference/generate_fixtures.R
# Exact supplied 43B data used as base. Derived synthetic data only.
options(digits=17)
RNGkind("Mersenne-Twister", "Inversion", "Rejection")
dir.create("reference/output", recursive=TRUE, showWarnings=FALSE)
d <- read.csv("fixtures/Test43B.csv")
stopifnot(nrow(d)==960L, length(unique(d$ID))==160L,
          length(unique(d$SCHOOL))==20L, !anyNA(d),
          identical(sort(unique(d$TIME)), 0:5) || all(sort(unique(d$TIME))==0:5),
          !anyDuplicated(d[c("SCHOOL","ID","TIME")]))
write_fixture <- function(x, name) write.csv(x, file.path("reference/output",paste0(name,".csv")),row.names=FALSE,na="")
r <- d; r$ID <- ((r$ID-1) %% 8)+1
stopifnot(length(unique(r$ID))==8, nrow(unique(r[c("SCHOOL","ID")]))==160)
write_fixture(r,"Test45A_ReusedIDs")
i <- d; i$TIME <- c(0,1,2,4,7,10)[i$TIME+1L]
write_fixture(i,"Test46A_IrregularTime")
m <- i; m$Y[m$TIME==2 | (m$TIME==7 & m$ID %in% head(sort(unique(m$ID)),40))] <- NA
stopifnot(sum(is.na(m$Y))==200)
write_fixture(m,"Test46B_MissingTime")
write_fixture(rbind(i,i[1,]),"Test46C_DuplicateTime")
write_fixture(d[d$TIME<=4,],"Test50C_Filtered")
# Test 49A: detrended simulated residuals deliberately not iid; stress test only.
set.seed(4901)
s <- d; s$SCHOOL <- factor(s$SCHOOL); s$ID <- factor(s$ID); s$GROUP <- factor(s$GROUP)
school_re <- rnorm(20,0,2); id_re <- rnorm(160,0,1.5); e <- rnorm(nrow(s))
for (j in split(seq_len(nrow(s)),s$ID)) e[j] <- residuals(lm(e[j]~s$TIME[j]))
s$Y <- 20+.5*s$TIME+school_re[as.integer(s$SCHOOL)]+id_re[as.integer(s$ID)]+e
write_fixture(s,"Test49A_Singularity")
# Identity adversary: disjoint samples, same likelihood under linear regression.
a <- data.frame(ROW=1:80, X=rep(1:40,2), S=rep(c("A","B"),each=40))
set.seed(13301); eps <- rnorm(40); a$Y <- c(50+2*(1:40)+eps,72-2*(1:40)-eps)
write_fixture(a,"Identity_DisjointEqualFit")
# Arbitrary interaction subset + >=3 level factor to test custom editor and recoding.
c <- d; c$Z <- (c$ID %% 3); c$W <- sin(c$ID)+cos(c$TIME); c$Y <- c$Y+.3*c$TIME*c$Z+.15*c$W
write_fixture(c,"CustomTerms")
cat("Synthetic fixture generation complete.\n")
