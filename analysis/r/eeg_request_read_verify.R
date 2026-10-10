# Independent source-file reader, discrete PSD integration, and full-family BH.
# Does not consume Python's parsed data or production calculation helpers.
runtime_arg <- grep('^--file=', commandArgs(), value=TRUE)[1]
runtime_dir <- dirname(sub('^--file=', '', runtime_arg))
source(file.path(runtime_dir,'runtime_profile.R'))
configure_locked_runtime(file.path(runtime_dir,'runtime-versions.lock.json'),'independent')
args <- commandArgs(TRUE)
job <- jsonlite::fromJSON(args[[1]], simplifyVector=FALSE)
out <- job$output
dir.create(file.path(out,"r_reads"),recursive=TRUE)
dir.create(file.path(out,"r_bh"),recursive=TRUE)
read_text_csv <- function(path) {
  x <- withCallingHandlers(read.csv(path,check.names=FALSE,colClasses="character",
    na.strings=NULL,encoding="UTF-8",stringsAsFactors=FALSE),warning=function(w) stop(w))
  names(x)[[1]] <- sub("^\ufeff","",names(x)[[1]])
  x
}
write_utf8 <- function(x,path) write.csv(x,path,row.names=FALSE,na="",fileEncoding="UTF-8")
for (record in job$reads) write_utf8(read_text_csv(record$path),file.path(out,"r_reads",paste0(record$id,".csv")))
q <- as.data.frame(readxl::read_excel(job$questionnaire_file,sheet=1,col_types="text",.name_repair="minimal"),check.names=FALSE)
columns <- unlist(job$questionnaire_columns)
q <- q[,columns,drop=FALSE]; q[is.na(q)] <- ""
write_utf8(q,file.path(out,"r_questionnaire.csv"))
answer <- q[[2]]
group <- ifelse(grepl("从不|极少|偶尔",answer),"Low",ifelse(grepl("有时|经常",answer),"High","Unknown"))
identity <- vapply(strsplit(trimws(q[[1]]),"[·•]"),function(x) {x<-trimws(x);x<-x[nzchar(x)];if(length(x)) x[[1]] else ""},character(1))
groups <- data.frame(Participant=identity,answer=answer,ExperienceGroup=group,check.names=FALSE)
names(groups)[2] <- "Q1.4"
write_utf8(groups,file.path(out,"r_Q1_4_groups.csv"))

field <- function(x,name) {
  labels <- if(!is.null(names(x))) names(x) else dimnames(x)[[1]]
  index <- match(name,labels)
  if(is.na(index)) stop(paste("Missing MAT field",name))
  x[[index]]
}
paired <- read_text_csv(file.path(job$cache$cache_dir,"paired_powers.csv"))
rows <- list(); n <- 0L
bounds <- list(total_1_45=c(1,45),total_1_40=c(1,40),power_40_45=c(40,45),theta=c(4,7),alpha=c(8,12),beta=c(13,30))
for (i in seq_along(job$cache$sources)) {
  source <- job$cache$sources[[i]]
  x <- R.matlab::readMat(file.path(job$cache$cache_dir,sprintf("spectra_%03d.mat",i)))
  person <- as.character(field(field(x$metadata,"source"),"participant"))
  if(person != source$participant) stop("MAT source identity mismatch")
  for (r in 1:3) {
    roi <- c("F","P","O")[[r]]
    sub <- paired[paired$Participant==person & paired$roi==roi,,drop=FALSE]
    if(nrow(sub)!=dim(x$spectra)[[1]]) stop("PSD trial dimensions mismatch")
    for (j in seq_len(nrow(sub))) {
      s <- x$spectra[[j+(r-1)*nrow(sub)]][[1]]
      f <- as.numeric(field(s,"f")); pxx <- as.numeric(field(s,"pxx"))
      if(as.numeric(field(s,"first.sample"))!=as.numeric(sub$first_sample[[j]]) ||
         as.numeric(field(s,"last.sample"))!=as.numeric(sub$last_sample[[j]])) stop("PSD trial endpoint mismatch")
      if(any(!is.finite(pxx)) || any(pxx<0) || any(diff(f)<=0)) stop("Invalid PSD")
      row <- sub[j,c("Participant","GlobalTrialOrder","onset_trim_s","roi"),drop=FALSE]
      for (name in names(bounds)) {
        mask <- f>=bounds[[name]][[1]] & f<=bounds[[name]][[2]]
        ff <- f[mask]; pp <- pxx[mask]
        row[[name]] <- sum(diff(ff)*(head(pp,-1)+tail(pp,-1))/2)
      }
      n <- n+1L; rows[[n]] <- row
    }
  }
  cat("R PSD verified",i,"of",length(job$cache$sources),"\n")
}
write_utf8(do.call(rbind,rows),file.path(out,"r_psd_integrals.csv"))
versions_to_check <- if (is.null(job$versions)) c("A","B","C") else unlist(job$versions)
for (version in versions_to_check) for (kind in c("coefficients","factors")) {
  x <- read_text_csv(file.path(job$source_run,version,paste0("family_",kind,".csv")))
  x$p.value <- as.numeric(x$p.value)
  x$within_q <- x$joint_q <- NA_real_
  for (family in unique(x$family_id)) {
    all <- which(x$family_id==family)
    sets <- c(list(all),lapply(unique(x$onset_trim_s[all]),function(w) all[x$onset_trim_s[all]==w]))
    for (k in seq_along(sets)) {
      ids <- sets[[k]]; p <- x$p.value[ids]
      valid <- all(is.finite(p) & p>=0 & p<=1) && all(tolower(x$inference_valid[ids])=="true")
      if(valid) x[ids,if(k==1) "joint_q" else "within_q"] <- p.adjust(p,method="BH")
    }
  }
  write_utf8(x,file.path(out,"r_bh",paste0(version,"_",kind,".csv")))
}
packages <- c("readxl","R.matlab","digest","lme4","clubSandwich","jsonlite","lmerTest")
versions <- setNames(lapply(packages,function(p) as.character(utils::packageVersion(p))),packages)
jsonlite::write_json(list(R=R.version.string,packages=versions,library_paths=.libPaths(),
  source_reads="independent read.csv/readxl/R.matlab",spectra=n),file.path(out,"R_environment.json"),auto_unbox=TRUE,pretty=TRUE)
writeLines(capture.output(sessionInfo()),file.path(out,"R_session_info.txt"))
