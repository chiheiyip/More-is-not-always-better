# Read original XLSX independently; no Python conversion or production helpers.
suppressPackageStartupMessages({library(jsonlite);library(readxl)})
job<-fromJSON(commandArgs(TRUE)[1]);dir.create(job$outdir,recursive=TRUE,showWarnings=FALSE)
q<-as.data.frame(read_excel(job$questionnaire_file,sheet=1,col_names=TRUE,.name_repair="minimal"),check.names=FALSE)
field<-function(code){cols<-names(q)[grepl(paste0("^",gsub(".","\\.",code,fixed=TRUE),"(_|\\s|$)"),names(q),perl=TRUE)&!grepl("_word$",names(q))];if(length(cols)!=1)stop("Ambiguous original questionnaire field");cols}
canonical<-function(v)sub("-\\d{1,4}-\\d{1,4}$","",trimws(sub("[·•].*$","",v)),perl=TRUE)
names_original<-as.character(q[[field("Q1.0")]]);original<-as.character(q[[field("Q1.4")]])
group<-ifelse(grepl("^(从不|偶尔)",original),"Low",ifelse(grepl("^(有时|经常)",original),"High",NA_character_))
if(anyNA(group))stop("Unrecognized Q1.4 answer")
groups<-data.frame(Participant=canonical(names_original),Q1.4Original=original,ExerciseFrequency=group,QuestionnaireName=names_original,AuthoritativeField=field("Q1.4"))
if(anyDuplicated(groups$Participant))stop("Duplicate original questionnaire identity")
people<-as.data.frame(read_excel(job$participant_information,sheet=1,col_names=TRUE),check.names=FALSE)
design<-as.data.frame(read_excel(job$trial_order_mapping,sheet=1,col_names=TRUE),check.names=FALSE)
if(anyDuplicated(design[c("Participant","GlobalTrialOrder")]))stop("Duplicate original design key")
design<-design[order(design$Participant,design$GlobalTrialOrder),c("Participant","GlobalTrialOrder","WWR","Complexity","Block","PositionWithinBlock","OrderGroup","SceneID")]
design$PositionWithinBlockCentered<-design$PositionWithinBlock-3.5
design$PreviousWWR<-NA_real_;design$PreviousComplexity<-NA_real_
for(p in unique(design$Participant))for(b in c(1,2)) {
  ix<-which(design$Participant==p&design$Block==b)
  if(length(ix)!=6)stop("Original block does not contain six trials")
  design$PreviousWWR[ix[-1]]<-design$WWR[ix[-length(ix)]];design$PreviousComplexity[ix[-1]]<-design$Complexity[ix[-length(ix)]]
}
design<-merge(design,people[c("Participant","Gender")],by="Participant",all.x=TRUE,sort=FALSE)
design<-merge(design,groups[c("Participant","Q1.4Original","ExerciseFrequency")],by="Participant",all.x=TRUE,sort=FALSE)
if(anyNA(design[c("Gender","ExerciseFrequency")]))stop("Original participant linkage failed")
write.csv(groups,file.path(job$outdir,"R_original_Q1_4_groups.csv"),row.names=FALSE,na="",fileEncoding="UTF-8")
write.csv(design,file.path(job$outdir,"R_original_design_fields.csv"),row.names=FALSE,na="",fileEncoding="UTF-8")
write_json(list(R=R.version.string,readxl=as.character(packageVersion("readxl")),questionnaire_source=job$questionnaire_file,authoritative_field=field("Q1.4"),raw_participants=nrow(groups),raw_trials=nrow(design)),file.path(job$outdir,"R_source_reading_provenance.json"),auto_unbox=TRUE,pretty=TRUE)
