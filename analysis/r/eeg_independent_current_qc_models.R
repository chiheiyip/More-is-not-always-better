# Additional current-source QC sensitivity. Independent of production R helpers.
suppressPackageStartupMessages({library(jsonlite);library(lme4);library(clubSandwich)})
job<-fromJSON(commandArgs(TRUE)[1]);contract<-fromJSON(job$contract)
dir.create(job$outdir,recursive=TRUE,showWarnings=FALSE)
core<-c("O_theta_relative","F_theta_relative","O_alpha_relative","O_beta_relative")
relative<-as.vector(outer(c("F","P","O"),c("theta","alpha","beta"),function(a,b)paste0(a,"_",b,"_relative")))
absolute<-sub("_relative$","_absolute",relative);absolute<-paste0("log10_",absolute)
factor_terms<-c("WWR","Complexity","ExperienceGroup","WWR:Complexity","WWR:ExperienceGroup","Complexity:ExperienceGroup")
environment_terms<-c("WWRWWR45","WWRWWR75","ComplexityC1","ExperienceGroupLow","WWRWWR45:ComplexityC1","WWRWWR75:ComplexityC1","WWRWWR45:ExperienceGroupLow","WWRWWR75:ExperienceGroupLow","ComplexityC1:ExperienceGroupLow")
base<-"WWR*Complexity + WWR*ExperienceGroup + Complexity*ExperienceGroup + Gender"
rhs<-list(Model1=paste(base,"+ Block + PositionWithinBlockCentered + OrderGroup + (1|Participant)"),
          Model0=paste(base,"+ (1|Participant)"),PreviousScene=paste(base,"+ Block + PositionWithinBlockCentered + OrderGroup + PreviousWWR + PreviousComplexity + (1|Participant)"),
          Block1=paste(base,"+ PositionWithinBlockCentered + OrderGroup + (1|Participant)"))
coef_rows<-factor_rows<-matrix_rows<-diagnostics<-list()
constraints<-function(model,d) {
  grid<-expand.grid(WWR=levels(d$WWR),Complexity=levels(d$Complexity),ExperienceGroup=levels(d$ExperienceGroup),stringsAsFactors=FALSE)
  grid$Gender<-levels(d$Gender)[1];grid$OrderGroup<-levels(d$OrderGroup)[1];grid$Block<-1.5;grid$PositionWithinBlockCentered<-0
  for(n in c("WWR","Complexity","ExperienceGroup","Gender","OrderGroup"))grid[[n]]<-factor(grid[[n]],levels=levels(d[[n]]))
  X<-model.matrix(delete.response(terms(nobars(formula(model)))),grid)[,names(fixef(model)),drop=FALSE]
  avg<-function(keep)colMeans(X[keep,,drop=FALSE])
  one<-function(n){lev<-levels(d[[n]]);do.call(rbind,lapply(lev[-1],function(v)avg(grid[[n]]==v)-avg(grid[[n]]==lev[1])))}
  two<-function(a,b){aa<-levels(d[[a]]);bb<-levels(d[[b]]);do.call(rbind,lapply(aa[-1],function(x)do.call(rbind,lapply(bb[-1],function(y)avg(grid[[a]]==x&grid[[b]]==y)-avg(grid[[a]]==x&grid[[b]]==bb[1])-avg(grid[[a]]==aa[1]&grid[[b]]==y)+avg(grid[[a]]==aa[1]&grid[[b]]==bb[1])))))}
  list(WWR=one("WWR"),Complexity=one("Complexity"),ExperienceGroup=one("ExperienceGroup"),`WWR:Complexity`=two("WWR","Complexity"),`WWR:ExperienceGroup`=two("WWR","ExperienceGroup"),`Complexity:ExperienceGroup`=two("Complexity","ExperienceGroup"))
}
for(version in c("D_current_QC_1_45","E_current_QC_1_40"))for(trim in if(is.null(job$test_trim))c(0,5,10,15) else job$test_trim) {
  d<-read.csv(file.path(job$inputs,version,paste0("trim_",trim,"s"),"input.csv"),check.names=FALSE,fileEncoding="UTF-8-BOM",na.strings=c("","NA"))
  for(n in names(contract$factor_levels))d[[n]]<-factor(d[[n]],levels=contract$factor_levels[[n]])
  d$Participant<-factor(d$Participant)
  for(outcome in if(is.null(job$test_outcomes))c(relative,absolute) else job$test_outcomes)for(suite in if(outcome%in%core)names(rhs) else "Model1") {
    if(version=="E_current_QC_1_40"&&outcome%in%absolute) {
      for(collection in c("coef_rows","factor_rows","matrix_rows","diagnostics")) {
        all<-get(collection);selected<-Filter(function(x)x$version[1]=="D_current_QC_1_45"&&x$onset_trim_s[1]==trim&&x$outcome[1]==outcome,all)
        for(row in selected){row$version<-version;if(collection=="diagnostics")row$reuse<-"D absolute: identical sample/design/numerator";all[[length(all)+1]]<-row};assign(collection,all)
      }
      next
    }
    x<-d
    if(suite=="PreviousScene")x<-x[x$PositionWithinBlock>1,,drop=FALSE]
    if(suite=="Block1")x<-x[x$Block==1,,drop=FALSE]
    f<-as.formula(paste(outcome,"~",rhs[[suite]]));error<-"";warning<-character()
    if(any(!complete.cases(x[,all.vars(f)])))stop("Incomplete independent current-QC model input")
    model<-tryCatch(withCallingHandlers(lmer(f,x,REML=FALSE),warning=function(w){warning<<-c(warning,conditionMessage(w));invokeRestart("muffleWarning")}),error=function(e){error<<-conditionMessage(e);NULL})
    valid<-!is.null(model)&&!isSingular(model)&&is.null(model@optinfo$conv$lme4$messages)&&all(model@optinfo$conv$opt==0)
    id<-data.frame(version=version,onset_trim_s=trim,outcome=outcome,model=suite)
    diagnostics[[length(diagnostics)+1]]<-cbind(id,data.frame(participants=length(unique(x$Participant)),trials=nrow(x),inference_valid=valid,converged=!is.null(model)&&is.null(model@optinfo$conv$lme4$messages),singular=if(is.null(model))NA else isSingular(model),random_intercept_variance=if(is.null(model))NA else as.numeric(VarCorr(model)$Participant),warning=paste(warning,collapse="; "),error=error,reuse="new independent fit"))
    if(!is.null(model)) {
      V<-vcovCR(model,cluster=model.frame(model)$Participant,type="CR2");test<-coef_test(model,vcov=V,test="Satterthwaite");ci<-conf_int(model,vcov=V,test="Satterthwaite")
      coef_rows[[length(coef_rows)+1]]<-cbind(id,data.frame(term=rownames(test),estimate=test$beta,SE=sqrt(diag(vcov(model))),CR2_SE=test$SE,df=test$df_Satt,CI_low=ci$CI_L,CI_high=ci$CI_U,raw_p=test$p_Satt,inference_valid=valid))
      if(suite=="Model1")for(effect in factor_terms) {
        L<-constraints(model,x)[[effect]];wt<-Wald_test(model,constraints=L,vcov=V,test="HTZ")
        factor_rows[[length(factor_rows)+1]]<-cbind(id,data.frame(term=effect,estimate=if(nrow(L)==1)as.numeric(L%*%fixef(model)) else NA_real_,Fstat=wt$Fstat,df_num=wt$df_num,df_denom=wt$df_denom,raw_p=wt$p_val,inference_valid=valid))
        mat<-expand.grid(contrast_row=seq_len(nrow(L)),coefficient=colnames(L),stringsAsFactors=FALSE);mat$weight<-as.vector(L);mat$term<-effect
        matrix_rows[[length(matrix_rows)+1]]<-cbind(id,mat)
      }
    }
  }
  cat("Independent current-QC EEG:",version,trim,"\n")
}
co<-do.call(rbind,coef_rows);fa<-do.call(rbind,factor_rows)
family<-function(source,name,outcomes,terms,suite="Model1") {
  skeleton<-expand.grid(version=c("D_current_QC_1_45","E_current_QC_1_40"),onset_trim_s=c(0,5,10,15),outcome=outcomes,term=terms,model=suite,stringsAsFactors=FALSE)
  value<-merge(skeleton,source,by=c("version","onset_trim_s","outcome","term","model"),all.x=TRUE,sort=FALSE)
  value$family_id<-name;value$within_q<-NA_real_;value$joint_q<-NA_real_
  for(v in unique(value$version)) {
    ids<-which(value$version==v)
    if(all(value$inference_valid[ids]%in%TRUE)&&all(is.finite(value$raw_p[ids])))value$joint_q[ids]<-p.adjust(value$raw_p[ids],"BH",n=length(ids))
    for(t in c(0,5,10,15)) {
      j<-ids[value$onset_trim_s[ids]==t]
      if(all(value$inference_valid[j]%in%TRUE)&&all(is.finite(value$raw_p[j])))value$within_q[j]<-p.adjust(value$raw_p[j],"BH",n=length(j))
    }
  }
  value
}
families<-list(family(co,"coefficient_core_relative",core,environment_terms),family(co,"coefficient_expanded_relative",relative,environment_terms),family(co,"coefficient_expanded_absolute",absolute,environment_terms),family(co,"temporal_relative",relative,c("Block","PositionWithinBlockCentered")),family(co,"temporal_absolute",absolute,c("Block","PositionWithinBlockCentered")),family(co,"previous_core_relative",core,c("PreviousWWRWWR45","PreviousWWRWWR75","PreviousComplexityC1"),"PreviousScene"))
factors<-list(family(fa,"factor_core_relative",core,factor_terms),family(fa,"factor_expanded_relative",relative,factor_terms),family(fa,"factor_expanded_absolute",absolute,factor_terms))
for(pair in list(list(co,"current_QC_coefficients.csv"),list(fa,"current_QC_factor_tests.csv"),list(do.call(rbind,diagnostics),"current_QC_diagnostics.csv"),list(do.call(rbind,matrix_rows),"current_QC_contrast_matrices.csv"),list(do.call(rbind,families),"current_QC_coefficient_families.csv"),list(do.call(rbind,factors),"current_QC_factor_families.csv")))write.csv(pair[[1]],file.path(job$outdir,pair[[2]]),row.names=FALSE,na="",fileEncoding="UTF-8")
write_json(list(R=R.version.string,lme4=as.character(packageVersion("lme4")),clubSandwich=as.character(packageVersion("clubSandwich")),versions="Current-source QC sensitivities D/E; historical A/B/C preserved"),file.path(job$outdir,"R_current_QC_environment.json"),auto_unbox=TRUE,pretty=TRUE)
