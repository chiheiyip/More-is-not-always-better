# Direct lme4/clubSandwich refits. Marginal constraints are independently built
# from an equal-weight prediction grid, not production's hand-coded matrices.
invisible(Sys.setlocale("LC_CTYPE", ".UTF-8"))
job <- jsonlite::fromJSON(commandArgs(TRUE)[[1]])
design <- jsonlite::fromJSON(job$design)
requests <- read.csv(job$requests)
coef_rows <- factor_rows <- matrix_rows <- diagnostic_rows <- list()
design_rows <- list()
read_input <- function(path) {
  x <- read.csv(path,check.names=FALSE,encoding="UTF-8",na.strings=c("","NA"))
  names(x)[[1]] <- sub("^\ufeff","",names(x)[[1]])
  for (name in names(design$factor_levels)) {
    x[[name]] <- factor(x[[name]],levels=design$factor_levels[[name]])
    contrasts(x[[name]]) <- contr.treatment(levels(x[[name]]),base=1)
  }
  x$Participant <- factor(x$Participant)
  x
}
base <- "WWR*Complexity + WWR*ExperienceGroup + Complexity*ExperienceGroup + Gender"
rhs <- list(Model1=paste(base,"+ Block + PositionWithinBlockCentered + OrderGroup + (1|Participant)"),
            Model0=paste(base,"+ (1|Participant)"),
            PreviousScene=paste(base,"+ Block + PositionWithinBlockCentered + OrderGroup + PreviousWWR + PreviousComplexity + (1|Participant)"),
            Block1=paste(base,"+ PositionWithinBlockCentered + OrderGroup + (1|Participant)"))
grid_constraints <- function(model,data) {
  grid <- expand.grid(WWR=levels(data$WWR),Complexity=levels(data$Complexity),
    ExperienceGroup=levels(data$ExperienceGroup),Gender=levels(data$Gender),OrderGroup=levels(data$OrderGroup),stringsAsFactors=FALSE)
  grid$Block <- 1.5; grid$PositionWithinBlockCentered <- 0
  for (name in c("WWR","Complexity","ExperienceGroup","Gender","OrderGroup")) grid[[name]]<-factor(grid[[name]],levels=levels(data[[name]]))
  formula <- lme4::nobars(formula(model)); formula[[2]]<-NULL
  X <- model.matrix(formula,grid,contrasts.arg=lapply(data[c("WWR","Complexity","ExperienceGroup","Gender","OrderGroup")],contrasts))
  X <- X[,names(lme4::fixef(model)),drop=FALSE]
  meanrow <- function(ids) colMeans(X[ids,,drop=FALSE])
  main <- function(name) {
    lev<-levels(data[[name]])
    do.call(rbind,lapply(lev[-1],function(l) meanrow(grid[[name]]==l)-meanrow(grid[[name]]==lev[[1]])))
  }
  interaction <- function(a,b) {
    aa<-levels(data[[a]]); bb<-levels(data[[b]])
    do.call(rbind,lapply(aa[-1],function(x) do.call(rbind,lapply(bb[-1],function(y)
      meanrow(grid[[a]]==x & grid[[b]]==y)-meanrow(grid[[a]]==x & grid[[b]]==bb[[1]])-
      meanrow(grid[[a]]==aa[[1]] & grid[[b]]==y)+meanrow(grid[[a]]==aa[[1]] & grid[[b]]==bb[[1]])))))
  }
  list(WWR=main("WWR"),Complexity=main("Complexity"),ExperienceGroup=main("ExperienceGroup"),
    "WWR:Complexity"=interaction("WWR","Complexity"),"WWR:ExperienceGroup"=interaction("WWR","ExperienceGroup"),
    "Complexity:ExperienceGroup"=interaction("Complexity","ExperienceGroup"))
}
for (version in c("A","B","C")) for (window in unique(requests$onset_trim_s)) {
  input<-read_input(file.path(job$source_run,version,paste0("trim_",window,"s"),"input.csv"))
  selected<-requests[requests$onset_trim_s==window,,drop=FALSE]
  for(i in seq_len(nrow(selected))) {
    outcome<-selected$outcome[[i]]; suite<-selected$model[[i]]
    if(is.null(rhs[[suite]])) stop("Unknown model")
    data<-input
    if(suite=="PreviousScene") data<-data[data$PositionWithinBlock>1,,drop=FALSE]
    if(suite=="Block1") data<-data[data$Block==1,,drop=FALSE]
    formula<-as.formula(paste(outcome,"~",rhs[[suite]]))
    if(any(!complete.cases(data[,all.vars(formula),drop=FALSE]))) stop("Incomplete cited model sample")
    model<-lme4::lmer(formula,data,REML=FALSE)
    if(lme4::isSingular(model) || length(model@optinfo$conv$lme4$messages)>0 || any(model@optinfo$conv$opt!=0)) stop("Invalid refit diagnostic")
    V<-clubSandwich::vcovCR(model,type="CR2",cluster=model.frame(model)$Participant)
    xx<-as.data.frame(lme4::getME(model,"X"),check.names=FALSE)
    design_rows[[length(design_rows)+1]]<-cbind(data.frame(version=version,onset_trim_s=window,outcome=outcome,model=suite,
      Participant=data$Participant,GlobalTrialOrder=data$GlobalTrialOrder),xx)
    test<-clubSandwich::coef_test(model,vcov=V,test="Satterthwaite")
    ci<-clubSandwich::conf_int(model,vcov=V,test="Satterthwaite")
    row<-data.frame(version=version,onset_trim_s=window,outcome=outcome,model=suite,term=rownames(test),
      estimate=test$beta,std.error=test$SE,df=test$df_Satt,p.value=test$p_Satt,CI_low=ci$CI_L,CI_high=ci$CI_U)
    coef_rows[[length(coef_rows)+1]]<-row
    diagnostic_rows[[length(diagnostic_rows)+1]]<-data.frame(version=version,onset_trim_s=window,outcome=outcome,model=suite,n_trials=nobs(model),n_participants=length(unique(data$Participant)),fit="valid")
    if(suite=="Model1") {
      constraints<-grid_constraints(model,data)
      for(effect in names(constraints)) {
        L<-constraints[[effect]]
        stat<-clubSandwich::Wald_test(model,constraints=L,vcov=V,test="HTZ")
        estimate<-if(nrow(L)==1) drop(L%*%lme4::fixef(model)) else NA_real_
        factor_rows[[length(factor_rows)+1]]<-data.frame(version=version,onset_trim_s=window,outcome=outcome,model=suite,term=effect,
          estimate=estimate,Fstat=stat$Fstat,df_num=stat$df_num,df_denom=stat$df_denom,p.value=stat$p_val)
        matrix<-expand.grid(contrast_row=seq_len(nrow(L)),coefficient=colnames(L),stringsAsFactors=FALSE)
        matrix$weight<-as.vector(L);matrix$version<-version;matrix$onset_trim_s<-window;matrix$outcome<-outcome;matrix$term<-effect
        matrix_rows[[length(matrix_rows)+1]]<-matrix
      }
    }
    cat("Refit",version,window,outcome,suite,"\n")
  }
}
for(name in c("coefficients","factors","matrices","diagnostics")) {
  value<-switch(name,coefficients=coef_rows,factors=factor_rows,matrices=matrix_rows,diagnostics=diagnostic_rows)
  write.csv(do.call(rbind,value),file.path(job$output,paste0("r_refit_",name,".csv")),row.names=FALSE,na="",fileEncoding="UTF-8")
}
jsonlite::write_json(list(R=R.version.string,lme4=as.character(packageVersion("lme4")),
  clubSandwich=as.character(packageVersion("clubSandwich"))),file.path(job$output,"model_R_environment.json"),auto_unbox=TRUE,pretty=TRUE)
all_columns<-unique(unlist(lapply(design_rows,names)))
design_rows<-lapply(design_rows,function(x) {for(name in setdiff(all_columns,names(x))) x[[name]]<-NA_real_;x[,all_columns,drop=FALSE]})
write.csv(do.call(rbind,design_rows),file.path(job$output,"r_refit_design.csv"),row.names=FALSE,na="",fileEncoding="UTF-8")
