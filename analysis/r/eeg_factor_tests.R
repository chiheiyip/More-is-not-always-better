# Reimplemented equal-weight marginal contrasts; CR2 covariance and HTZ tests.
# The recovered historical coefficient models do not contain this implementation.
eeg_factor_constraints <- function(coefficient_names) {
  blank <- function(n) matrix(0, n, length(coefficient_names), dimnames=list(NULL, coefficient_names))
  required <- c('WWRWWR45','WWRWWR75','ComplexityC1','ExperienceGroupLow',
    'WWRWWR45:ComplexityC1','WWRWWR75:ComplexityC1',
    'WWRWWR45:ExperienceGroupLow','WWRWWR75:ExperienceGroupLow','ComplexityC1:ExperienceGroupLow')
  if (!all(required %in% coefficient_names)) stop('Missing coefficients for complete marginal contrasts')
  W <- blank(2)
  for (j in 1:2) {
    term <- c('WWRWWR45','WWRWWR75')[j]
    W[j,term] <- 1; W[j,paste0(term,':ComplexityC1')] <- .5
    W[j,paste0(term,':ExperienceGroupLow')] <- .5
  }
  C <- blank(1); C[1,'ComplexityC1'] <- 1
  C[1,c('WWRWWR45:ComplexityC1','WWRWWR75:ComplexityC1')] <- 1/3
  C[1,'ComplexityC1:ExperienceGroupLow'] <- .5
  E <- blank(1); E[1,'ExperienceGroupLow'] <- 1
  E[1,c('WWRWWR45:ExperienceGroupLow','WWRWWR75:ExperienceGroupLow')] <- 1/3
  E[1,'ComplexityC1:ExperienceGroupLow'] <- .5
  WC <- blank(2); WE <- blank(2); CE <- blank(1)
  for (j in 1:2) {
    WC[j,paste0(c('WWRWWR45','WWRWWR75')[j],':ComplexityC1')] <- 1
    WE[j,paste0(c('WWRWWR45','WWRWWR75')[j],':ExperienceGroupLow')] <- 1
  }
  CE[1,'ComplexityC1:ExperienceGroupLow'] <- 1
  list(WWR=W, Complexity=C, ExperienceGroup=E, 'WWR:Complexity'=WC,
       'WWR:ExperienceGroup'=WE, 'Complexity:ExperienceGroup'=CE)
}

eeg_factor_tests <- function(model, V, outcome, scale) {
  b <- lme4::fixef(model); constraints <- eeg_factor_constraints(names(b))
  rows <- list(); matrices <- list()
  for (effect in names(constraints)) {
    L <- constraints[[effect]]
    test <- as.data.frame(clubSandwich::Wald_test(model, constraints=L, vcov=V, test='HTZ'))
    se <- if(nrow(L)==1) sqrt(drop(L %*% V %*% t(L))) else NA_real_
    estimate <- if(nrow(L)==1) drop(L %*% b) else NA_real_
    margin <- if(nrow(L)==1) qt(.975, test$df_denom)*se else NA_real_
    rows[[effect]] <- data.frame(outcome=outcome, scale=scale, model='Model1', term=effect,
      estimate=estimate, std.error=se, CI_low=estimate-margin, CI_high=estimate+margin,
      Fstat=test$Fstat, df_num=test$df_num, df_denom=test$df_denom, p.value=test$p_val,
      n_trials=nobs(model), n_participants=length(unique(model.frame(model)$Participant)),
      method='reimplemented_equal_weight_marginal_CR2_HTZ')
    grid <- expand.grid(contrast_row=seq_len(nrow(L)), coefficient=colnames(L), stringsAsFactors=FALSE)
    grid$weight <- as.vector(L); grid$outcome <- outcome; grid$term <- effect
    matrices[[effect]] <- grid
  }
  list(tests=do.call(rbind,rows), matrices=do.call(rbind,matrices))
}
