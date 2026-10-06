# Independent eye modelling. Does not source any production eye/statistics helper.
suppressPackageStartupMessages({
  library(jsonlite); library(lme4); library(glmmTMB); library(clubSandwich); library(emmeans)
})
args <- commandArgs(trailingOnly=TRUE)
job <- fromJSON(args[1], simplifyVector=TRUE)
dir.create(job$outdir, recursive=TRUE, showWarnings=FALSE)
write_table <- function(x, name) write.csv(x, file.path(job$outdir,name), row.names=FALSE, na="", fileEncoding="UTF-8")
main <- read.csv(job$input, check.names=FALSE, stringsAsFactors=FALSE, fileEncoding="UTF-8-BOM", na.strings=c("","NA"))
identity_columns<-c("Participant","GlobalTrialOrder",if(job$mode=="boundary")"AuditVariant")
if(anyDuplicated(main[identity_columns]))stop("Duplicate independent trial/variant key")
encode <- function(d) {
  d$WWR <- factor(d$WWR,levels=c(15,45,75))
  d$Complexity <- factor(ifelse(as.character(d$Complexity)%in%c("0","C0"),"C0","C1"),levels=c("C0","C1"))
  d$ExerciseFrequency <- factor(d$ExerciseFrequency,levels=c("High","Low"))
  d$Gender <- factor(d$Gender,levels=c("Female","Male"))
  d$OrderGroup <- factor(d$OrderGroup,levels=c("new order2","order1","order2"))
  d$Participant <- factor(d$Participant)
  if("PreviousWWR"%in%names(d))d$PreviousWWR<-factor(d$PreviousWWR,levels=c(15,45,75))
  if("PreviousComplexity"%in%names(d))d$PreviousComplexity<-factor(ifelse(is.na(d$PreviousComplexity),NA,ifelse(as.character(d$PreviousComplexity)%in%c("0","C0"),"C0","C1")),levels=c("C0","C1"))
  d
}
core <- c("TableShare","WindowShare","RawCompetition","LogTableEnrichment","LogWindowEnrichment","AdjustedCompetition")
effects <- c("WWR","Complexity","ExerciseFrequency","WWR:Complexity","WWR:ExerciseFrequency","Complexity:ExerciseFrequency")
specs <- list()
add <- function(variant, outcome, kind, filter="QC60", omitted="", random_slope=FALSE) {
  specs[[length(specs)+1]] <<- list(variant=variant,outcome=outcome,kind=kind,filter=filter,omitted=omitted,random_slope=random_slope)
}
if(job$mode=="main") {
  for(outcome in core) {
    kind <- if(outcome%in%c("TableShare","WindowShare"))"ordered_beta" else "lmm"
    add("main",outcome,kind)
    for(cut in c("QC50","QC70"))add(tolower(cut),outcome,kind,cut)
    for(block in c("block1","block2"))add(block,outcome,kind)
    add("random_slope",outcome,kind,random_slope=TRUE)
    add("previous_scene",outcome,kind)
    if("SourceIdentityStatus"%in%names(main)&&any(grepl("unverified",main$SourceIdentityStatus)))add("exclude_unverified_identity",outcome,kind)
    if(kind=="ordered_beta") {add("logit_lmm",outcome,"logit_lmm");add("CR2_companion",outcome,"lmm");add("conditional_beta",outcome,"beta")}
  }
  for(aoi in c("Table","Window","Equipment","Background")) {
    add("secondary",paste0(aoi,"Visited"),"binomial")
    add("secondary",paste0(aoi,"FixationCount"),"negative_binomial")
    add("secondary",paste0(aoi,"TTFF"),"log_ttff")
  }
  for(o in c("CompositionTableRest","CompositionWindowRest","CompositionTableWindow"))add("composition",o,"lmm")
} else if(job$mode=="boundary") {
  for(v in unique(main$AuditVariant))for(o in core)add(v,o,if(o%in%c("TableShare","WindowShare"))"ordered_beta" else "lmm")
} else if(job$mode=="lopo") {
  for(p in job$participants)for(o in core)add("leave_one_participant_out",o,if(o%in%c("TableShare","WindowShare"))"ordered_beta" else "lmm",omitted=p)
} else if(job$mode=="test") {
  for(o in core)add("main",o,if(o%in%c("TableShare","WindowShare"))"beta" else "lmm")
} else if(job$mode=="ordered_test") {
  for(o in core)add("main",o,if(o%in%c("TableShare","WindowShare"))"ordered_beta" else "lmm")
} else stop("Unknown independent eye mode")

coefficients <- list(); factors <- list(); contrasts <- list(); matrices <- list(); diagnostics <- list()
fixed_rhs <- "WWR*Complexity + WWR*ExerciseFrequency + Complexity*ExerciseFrequency + Gender + Block + PositionWithinBlockCentered + OrderGroup"
grid_contrasts <- function(m, d) {
  grid <- expand.grid(WWR=factor(c(15,45,75),levels=c(15,45,75)),Complexity=factor(c("C0","C1"),levels=c("C0","C1")),ExerciseFrequency=factor(c("High","Low"),levels=c("High","Low")))
  grid$Gender <- factor("Female",levels=c("Female","Male"));grid$Block<-1;grid$PositionWithinBlockCentered<-0
  grid$OrderGroup<-factor("new order2",levels=c("new order2","order1","order2"));grid$Participant<-d$Participant[1]
  if("PreviousWWR"%in%all.vars(formula(m)))grid$PreviousWWR<-factor(15,levels=c(15,45,75))
  if("PreviousComplexity"%in%all.vars(formula(m)))grid$PreviousComplexity<-factor("C0",levels=c("C0","C1"))
  tt<-delete.response(terms(lme4::nobars(formula(m))))
  X<-model.matrix(tt,grid); nm<-names(fixef(m));X<-X[,nm,drop=FALSE]
  avg<-function(w=NULL,c=NULL,e=NULL){keep<-rep(TRUE,nrow(grid));if(!is.null(w))keep<-keep&grid$WWR==w;if(!is.null(c))keep<-keep&grid$Complexity==c;if(!is.null(e))keep<-keep&grid$ExerciseFrequency==e;colMeans(X[keep,,drop=FALSE])}
  L<-list(WWR=rbind(avg(45)-avg(15),avg(75)-avg(15)),
    Complexity=rbind(avg(c="C0")-avg(c="C1")),ExerciseFrequency=rbind(avg(e="High")-avg(e="Low")),
    `WWR:Complexity`=rbind((avg(45,"C0")-avg(15,"C0"))-(avg(45,"C1")-avg(15,"C1")),(avg(75,"C0")-avg(15,"C0"))-(avg(75,"C1")-avg(15,"C1"))),
    `WWR:ExerciseFrequency`=rbind((avg(45,e="High")-avg(15,e="High"))-(avg(45,e="Low")-avg(15,e="Low")),(avg(75,e="High")-avg(15,e="High"))-(avg(75,e="Low")-avg(15,e="Low"))),
    `Complexity:ExerciseFrequency`=rbind((avg(c="C0",e="High")-avg(c="C1",e="High"))-(avg(c="C0",e="Low")-avg(c="C1",e="Low"))))
  list(L=L, pairs=rbind(`WWR45-WWR15`=avg(45)-avg(15),`WWR75-WWR15`=avg(75)-avg(15),`WWR75-WWR45`=avg(75)-avg(45),`C0-C1`=avg(c="C0")-avg(c="C1")))
}
empty_coefficient <- function(id, status) cbind(id,data.frame(term=NA_character_,estimate=NA_real_,SE=NA_real_,CR2_SE=NA_real_,df=NA_real_,CI_low=NA_real_,CI_high=NA_real_,raw_p=NA_real_,inference=status))
likelihood_pairs <- function(m,id,f) {
  result<-list()
  for(factor in c("WWR","Complexity"))if(factor%in%all.vars(f)) {
    em<-tryCatch(emmeans(m,as.formula(paste("~",factor)),weights="equal"),error=function(e)NULL)
    if(!is.null(em)) {
      custom<-if(factor=="WWR")list(`WWR45-WWR15`=c(-1,1,0),`WWR75-WWR15`=c(-1,0,1),`WWR75-WWR45`=c(0,-1,1)) else list(`C0-C1`=c(1,-1))
      ps<-as.data.frame(summary(contrast(em,method=custom,adjust="none"),infer=c(TRUE,TRUE)))
      lower<-if("lower.CL"%in%names(ps))ps$lower.CL else ps$asymp.LCL;upper<-if("upper.CL"%in%names(ps))ps$upper.CL else ps$asymp.UCL
      result[[length(result)+1]]<-cbind(id,data.frame(contrast=ps$contrast,estimate=ps$estimate,SE=ps$SE,df=ps$df,CI_low=lower,CI_high=upper,raw_p=ps$p.value,inference="equal_weight_primary_likelihood"))
    }
  }
  result
}

for(i in seq_along(specs)) {
  s<-specs[[i]]; d<-main[main[[s$filter]]%in%c(TRUE,"True","TRUE",1),,drop=FALSE]
  if(s$variant%in%c("block1","block2"))d<-d[d$Block==if(s$variant=="block1")1 else 2,,drop=FALSE]
  if(job$mode=="boundary")d<-d[d$AuditVariant==s$variant,,drop=FALSE]
  if(nchar(s$omitted))d<-d[d$Participant!=s$omitted,,drop=FALSE]
  if(s$variant=="exclude_unverified_identity")d<-d[!grepl("unverified",d$SourceIdentityStatus),,drop=FALSE]
  if(startsWith(s$outcome,"Equipment"))d<-d[d$Complexity%in%c(1,"C1"),,drop=FALSE]
  d<-d[is.finite(d[[s$outcome]]),,drop=FALSE]
  zero <- if(nrow(d))mean(d[[s$outcome]]==0) else NA_real_
  one <- if(nrow(d))mean(d[[s$outcome]]==1) else NA_real_
  part <- "full"
  if(s$kind=="beta") {
    # Beta likelihood uses positive interior part when zeros occur. No arbitrary
    # epsilon or Smithson conversion; the corresponding Visited GLMM is separate.
    if(any(d[[s$outcome]]<=0|d[[s$outcome]]>=1))part<-"conditional_interior"
    d<-d[d[[s$outcome]]>0&d[[s$outcome]]<1,,drop=FALSE]
  }
  if(s$kind=="logit_lmm") {
    d<-d[d[[s$outcome]]>0,,drop=FALSE]
    d<-d[d[[s$outcome]]<1,,drop=FALSE]
  }
  if(s$kind=="log_ttff")d<-d[d[[s$outcome]]>=0,,drop=FALSE]
  if(s$variant=="previous_scene")d<-d[!is.na(d$PreviousWWR)&!is.na(d$PreviousComplexity),,drop=FALSE]
  d<-encode(d);d$Y<-d[[s$outcome]]
  if(s$kind=="logit_lmm")d$Y<-qlogis(d$Y)
  if(s$kind=="log_ttff")d$Y<-log1p(d$Y)
  rhs<-fixed_rhs
  if(s$variant%in%c("block1","block2"))rhs<-sub(" + Block", "", rhs,fixed=TRUE)
  if(startsWith(s$outcome,"Equipment"))rhs<-"WWR*ExerciseFrequency + Gender + Block + PositionWithinBlockCentered + OrderGroup"
  if(s$variant=="previous_scene")rhs<-paste(rhs,"+ PreviousWWR + PreviousComplexity")
  random<-if(s$random_slope)"(1 + WWR + Complexity | Participant)" else "(1 | Participant)"
  f<-as.formula(paste("Y ~",rhs,"+",random))
  if(s$kind=="negative_binomial")f<-update(f,.~.+offset(log(ValidTrackingSeconds)))
  id<-data.frame(variant=s$variant,outcome=s$outcome,model_kind=s$kind,omitted_participant=s$omitted,part=part,stringsAsFactors=FALSE)
  warnings<-character();error<-"";m<-NULL
  if(nrow(d)>20&&length(unique(d$Participant))>3&&length(unique(d$Y))>1) {
    m<-tryCatch(withCallingHandlers({
      if(s$kind%in%c("lmm","logit_lmm","log_ttff"))lmer(f,d,REML=FALSE,control=lmerControl(optimizer="bobyqa",optCtrl=list(maxfun=200000)))
      else glmmTMB(f,data=d,family=switch(s$kind,beta=beta_family(link="logit"),ordered_beta=ordbeta(link="logit"),binomial=binomial(link="logit"),negative_binomial=nbinom2(link="log")),control=glmmTMBControl(optCtrl=list(iter.max=10000,eval.max=10000)))
    },warning=function(w){warnings<<-c(warnings,conditionMessage(w));invokeRestart("muffleWarning")}),error=function(e){error<<-conditionMessage(e);NULL})
  } else error<-"insufficient sample or constant outcome"
  conv<-FALSE; singular<-NA;variance<-NA_real_;method<-"fit_failed"
  if(!is.null(m)) {
    if(inherits(m,"merMod")) {
      conv<-is.null(m@optinfo$conv$lme4$messages);singular<-isSingular(m,tol=1e-4);variance<-as.numeric(VarCorr(m)$Participant[1,1])
      b<-fixef(m);ordinary<-sqrt(diag(vcov(m)));V<-tryCatch(vcovCR(m,cluster=d$Participant,type="CR2"),error=function(e){error<<-conditionMessage(e);NULL})
      if(!is.null(V)) {
        contrasts<-c(contrasts,likelihood_pairs(m,id,f))
        ct<-as.data.frame(coef_test(m,vcov=V,test="Satterthwaite"));se<-ct$SE;df<-ct$df_Satt;p<-ct$p_Satt;crit<-qt(.975,df)
        coefficients[[length(coefficients)+1]]<-cbind(id,data.frame(term=names(b),estimate=as.numeric(b),SE=ordinary,CR2_SE=se,df=df,CI_low=b-crit*se,CI_high=b+crit*se,raw_p=p,inference="CR2_Satterthwaite"))
        method<-"LMM_ML_CR2_Satterthwaite"
        if(!startsWith(s$outcome,"Equipment")) {
          cc<-grid_contrasts(m,d)
          for(effect in effects) {
            L<-cc$L[[effect]];wt<-as.data.frame(Wald_test(m,constraints=L,vcov=V,test="HTZ"))
            factors[[length(factors)+1]]<-cbind(id,data.frame(effect=effect,F=wt$Fstat,df_num=wt$df_num,df_den=wt$df_denom,raw_p=wt$p_val,inference="equal_weight_CR2_HTZ"))
            for(k in seq_len(nrow(L)))matrices[[length(matrices)+1]]<-cbind(id,data.frame(effect=effect,contrast_row=k,term=names(b),weight=as.numeric(L[k,])))
          }
          for(k in seq_len(nrow(cc$pairs))) {
            L<-cc$pairs[k,,drop=FALSE];wt<-as.data.frame(Wald_test(m,constraints=L,vcov=V,test="HTZ"));estimate<-as.numeric(L%*%b);se<-sqrt(as.numeric(L%*%V%*%t(L)));df<-wt$df_denom
            contrasts[[length(contrasts)+1]]<-cbind(id,data.frame(contrast=rownames(cc$pairs)[k],estimate=estimate,SE=se,df=df,CI_low=estimate-qt(.975,df)*se,CI_high=estimate+qt(.975,df)*se,raw_p=wt$p_val,inference="equal_weight_CR2_HTZ"))
          }
        }
      }
    } else {
      conv<-m$fit$convergence==0&&isTRUE(m$sdr$pdHess);variance<-as.numeric(VarCorr(m)$cond$Participant[1,1]);singular<-variance<1e-8
      co<-summary(m)$coefficients$cond;b<-co[,1];se<-co[,2]
      coefficients[[length(coefficients)+1]]<-cbind(id,data.frame(term=rownames(co),estimate=b,SE=se,CR2_SE=NA_real_,df=Inf,CI_low=b-1.95996398454005*se,CI_high=b+1.95996398454005*se,raw_p=co[,4],inference="GLMM_Wald_z"))
      method<-paste0(s$kind,"_GLMM_Wald")
      jt<-tryCatch(as.data.frame(joint_tests(m,weights="equal")),error=function(e){error<<-conditionMessage(e);NULL})
      if(!is.null(jt)) {
        names(jt)[1]<-"effect"
        for(effect in intersect(effects,jt$effect)) {
          row<-jt[jt$effect==effect,,drop=FALSE]
          factors[[length(factors)+1]]<-cbind(id,data.frame(effect=effect,F=row$F.ratio,df_num=row$df1,df_den=row$df2,raw_p=row$p.value,inference="equal_weight_GLMM_Wald"))
        }
      }
      for(factor in c("WWR","Complexity"))if(factor%in%all.vars(f)) {
        em<-tryCatch(emmeans(m,as.formula(paste("~",factor)),weights="equal"),error=function(e)NULL)
        if(!is.null(em)) {
          custom<-if(factor=="WWR")list(`WWR45-WWR15`=c(-1,1,0),`WWR75-WWR15`=c(-1,0,1),`WWR75-WWR45`=c(0,-1,1)) else list(`C0-C1`=c(1,-1))
          ps<-as.data.frame(summary(contrast(em,method=custom,adjust="none"),infer=c(TRUE,TRUE)))
          lower<-if("lower.CL"%in%names(ps))ps$lower.CL else ps$asymp.LCL;upper<-if("upper.CL"%in%names(ps))ps$upper.CL else ps$asymp.UCL
          contrasts[[length(contrasts)+1]]<-cbind(id,data.frame(contrast=ps$contrast,estimate=ps$estimate,SE=ps$SE,df=ps$df,CI_low=lower,CI_high=upper,raw_p=ps$p.value,inference="equal_weight_GLMM_Wald"))
        }
      }
    }
  }
  if(is.null(m)||method=="fit_failed")coefficients[[length(coefficients)+1]]<-empty_coefficient(id,"fit_failed")
  diagnostics[[length(diagnostics)+1]]<-cbind(id,data.frame(participants=length(unique(d$Participant)),trials=nrow(d),zero_fraction=zero,one_fraction=one,converged=conv,singular=singular,random_intercept_variance=variance,formula=paste(deparse(f),collapse=" "),method=method,cluster="Participant",warnings=paste(unique(warnings),collapse="; "),error=error))
  if(i%%12==0)cat("Independent eye models",i,"/",length(specs),"\n")
}
bind <- function(rows) if(length(rows))do.call(rbind,rows) else data.frame()
co<-bind(coefficients);fa<-bind(factors);pa<-bind(contrasts);di<-bind(diagnostics)
if(nrow(co)) {
  identity<-c("variant","outcome","model_kind","omitted_participant","part")
  co<-merge(co,di[c(identity,"converged")],by=identity,all.x=TRUE,sort=FALSE)
  # Companion estimates and CR2 inference stay together; do not attach a
  # Gaussian companion p to an ordered-beta coefficient or interval.
  if(job$mode=="main") {
    companions<-co[(co$variant=="CR2_companion"|co$variant=="main"&co$outcome=="RawCompetition"),,drop=FALSE]
    companions$variant<-"CR2_companion_family";co<-rbind(co,companions)
  }
  co$family<-ifelse(co$outcome%in%core[1:3],"A",ifelse(co$outcome%in%core[4:6],"B","supplementary"))
  co$q<-NA_real_;co$family_size<-NA_integer_
  for(v in unique(co$variant))for(term in unique(co$term))for(fam in c("A","B")) {
    if(is.na(term))next
    ix<-which(co$variant==v&co$term==term&co$family==fam);co$family_size[ix]<-3L
    for(p in unique(co$omitted_participant[ix])) {
      j<-ix[co$omitted_participant[ix]==p]
      expected<-if(fam=="A")core[1:3] else core[4:6]
      if(length(j)==3&&setequal(co$outcome[j],expected)&&all(is.finite(co$raw_p[j]))&&all(co$converged[j]))co$q[j]<-p.adjust(co$raw_p[j],method="BH",n=3)
    }
  }
}
if(nrow(fa)) {
  identity<-c("variant","outcome","model_kind","omitted_participant","part")
  fa<-merge(fa,di[c(identity,"converged")],by=identity,all.x=TRUE,sort=FALSE)
  fa$family<-ifelse(fa$outcome%in%core[1:3],"A",ifelse(fa$outcome%in%core[4:6],"B","supplementary"))
  fa$q<-NA_real_;fa$family_size<-NA_integer_
  for(v in unique(fa$variant))for(e in effects)for(fam in c("A","B")) {
    ix<-which(fa$variant==v&fa$effect==e&fa$family==fam);fa$family_size[ix]<-3L
    # Each LOPO omission is a separate sensitivity, never one giant p-value family.
    for(p in unique(fa$omitted_participant[ix])) {
      j<-ix[fa$omitted_participant[ix]==p]
      if(length(j)==3&&all(is.finite(fa$raw_p[j]))&&all(fa$converged[j]))fa$q[j]<-p.adjust(fa$raw_p[j],method="BH",n=3)
    }
  }
}
if(nrow(pa)) {
  pa$Holm_p<-NA_real_
  for(v in unique(pa$variant))for(o in unique(pa$outcome))for(p in unique(pa$omitted_participant))for(inference in unique(pa$inference)) {
    j<-which(pa$variant==v&pa$outcome==o&pa$omitted_participant==p&pa$inference==inference&startsWith(pa$contrast,"WWR"))
    if(length(j)==3&&all(is.finite(pa$raw_p[j])))pa$Holm_p[j]<-p.adjust(pa$raw_p[j],"holm",n=3)
  }
}
write_table(co,"independent_coefficients.csv");write_table(fa,"independent_factor_tests.csv")
write_table(pa,"independent_contrasts.csv");write_table(di,"08_model_specification_audit.csv")
write_table(bind(matrices),"independent_contrast_matrices.csv")
if(nrow(fa))write_table(fa,"09_multiplicity_audit.csv")
if(nrow(pa))write_table(pa,"10_contrast_direction_audit.csv")
env<-list(R=R.version.string,packages=lapply(c("lme4","glmmTMB","clubSandwich","emmeans","jsonlite"),function(p)as.character(packageVersion(p))),mode=job$mode,models=length(specs),independence="independent script, independent trial input; no production helpers/results read")
names(env$packages)<-c("lme4","glmmTMB","clubSandwich","emmeans","jsonlite")
write_json(env,file.path(job$outdir,"R_environment.json"),auto_unbox=TRUE,pretty=TRUE)
