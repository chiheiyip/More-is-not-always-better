args <- commandArgs(TRUE)
source('analysis/r/eeg_factor_tests.R')
library(lme4); library(clubSandwich)
d <- read.csv(args[1], encoding='UTF-8',check.names=FALSE); names(d)[1] <- sub('^\ufeff','',names(d)[1])
contract <- jsonlite::fromJSON(args[2])
for (field in names(contract$factor_levels)) {
  d[[field]] <- factor(d[[field]],levels=contract$factor_levels[[field]])
  contrasts(d[[field]]) <- contr.treatment(contract$factor_levels[[field]],base=1)
}
m <- lmer(F_theta_relative ~ WWR*Complexity + WWR*ExperienceGroup + Complexity*ExperienceGroup + Gender + Block + PositionWithinBlockCentered + OrderGroup + (1|Participant),d,REML=FALSE)
V <- vcovCR(m,cluster=model.frame(m)$Participant,type='CR2')
tests <- eeg_factor_tests(m,V,'F_theta_relative','relative_power')$tests
L <- eeg_factor_constraints(names(fixef(m)))
# Independent balanced marginal grid: not the helper's hard-coded weights.
grid <- expand.grid(WWR=levels(d$WWR),Complexity=levels(d$Complexity),ExperienceGroup=levels(d$ExperienceGroup),Gender=levels(d$Gender),OrderGroup=levels(d$OrderGroup))
grid$Block <- 1; grid$PositionWithinBlockCentered <- 0
X <- model.matrix(lme4::nobars(formula(m))[-2],grid,
  contrasts.arg=lapply(d[c('WWR','Complexity','ExperienceGroup','Gender','OrderGroup')],contrasts))
X <- X[,names(fixef(m))]
for (effect in c('WWR','Complexity','ExperienceGroup')) {
  groups <- lapply(split(seq_len(nrow(X)),factor(grid[[effect]],levels=levels(d[[effect]]))),function(i) colMeans(X[i,,drop=FALSE]))
  expected <- do.call(rbind,lapply(groups[-1],function(x) x-groups[[1]]))
  stopifnot(max(abs(expected-L[[effect]]))<1e-12)
}
for (i in seq_len(nrow(tests))) {
  row <- tests[i,]; w <- Wald_test(m,constraints=L[[row$term]],vcov=V,test='HTZ')
  stopifnot(abs(row$p.value-w$p_val)<1e-12)
  if (row$df_num>1) stopifnot(is.na(row$estimate),is.na(row$CI_low))
  else stopifnot(abs(row$Fstat-(row$estimate/row$std.error)^2)<1e-10)
}
stopifnot(inherits(try(eeg_factor_constraints('(Intercept)'),silent=TRUE),'try-error'))
cat('Independent equal-weight grid, CR2/HTZ, intervals and missing coefficients verified\n')
