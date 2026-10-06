# Add ordinary SE and fit diagnostics absent from prior independent audit outputs.
# No production statistical helper is sourced; prior results are read afterwards.
suppressPackageStartupMessages({library(jsonlite);library(lme4);library(clubSandwich)})
job<-fromJSON(commandArgs(TRUE)[1]);contract<-fromJSON(job$contract)
rows<-list();diagnostics<-list()
rhs<-"WWR*Complexity + WWR*ExperienceGroup + Complexity*ExperienceGroup + Gender + Block + PositionWithinBlockCentered + OrderGroup + (1|Participant)"
for(version in c("A","B","C"))for(trim in c(0,5,10,15)) {
  x<-read.csv(file.path(job$source_run,version,paste0("trim_",trim,"s"),"input.csv"),check.names=FALSE,fileEncoding="UTF-8-BOM",na.strings=c("","NA"))
  for(n in names(contract$factor_levels))x[[n]]<-factor(x[[n]],levels=contract$factor_levels[[n]])
  x$Participant<-factor(x$Participant)
  for(outcome in c("O_theta_relative","F_theta_relative","O_alpha_relative","O_beta_relative")) {
    formula<-as.formula(paste(outcome,"~",rhs))
    if(any(!complete.cases(x[,all.vars(formula)])))stop("Incomplete independent core EEG input")
    model<-lmer(formula,x,REML=FALSE)
    V<-vcovCR(model,cluster=model.frame(model)$Participant,type="CR2")
    test<-coef_test(model,vcov=V,test="Satterthwaite");ci<-conf_int(model,vcov=V,test="Satterthwaite")
    rows[[length(rows)+1]]<-data.frame(version=version,onset_trim_s=trim,outcome=outcome,model="Model1",term=rownames(test),estimate=test$beta,SE=sqrt(diag(vcov(model))),CR2_SE=test$SE,df=test$df_Satt,CI_low=ci$CI_L,CI_high=ci$CI_U,raw_p=test$p_Satt)
    diagnostics[[length(diagnostics)+1]]<-data.frame(version=version,onset_trim_s=trim,outcome=outcome,participants=length(unique(x$Participant)),trials=nobs(model),converged=is.null(model@optinfo$conv$lme4$messages)&&all(model@optinfo$conv$opt==0),singular=isSingular(model),random_intercept_variance=as.numeric(VarCorr(model)$Participant),residual_variance=sigma(model)^2,cluster="Participant")
  }
  cat("Independent core EEG details:",version,trim,"\n")
}
write.csv(do.call(rbind,rows),file.path(job$outdir,"independent_eeg_core_details.csv"),row.names=FALSE,na="",fileEncoding="UTF-8")
write.csv(do.call(rbind,diagnostics),file.path(job$outdir,"independent_eeg_core_diagnostics.csv"),row.names=FALSE,na="",fileEncoding="UTF-8")
write_json(list(R=R.version.string,lme4=as.character(packageVersion("lme4")),clubSandwich=as.character(packageVersion("clubSandwich")),models=length(diagnostics)),file.path(job$outdir,"R_details_environment.json"),auto_unbox=TRUE,pretty=TRUE)
