# Same Gaussian companion sensitivity used in eye_stage2_analysis.R, restricted
# to the revised exact common trial sample; eye-only main fits are unchanged.
args <- commandArgs(trailingOnly=TRUE)
file_arg <- grep("^--file=",commandArgs(FALSE),value=TRUE)
source(file.path(dirname(normalizePath(sub("^--file=","",file_arg[[1]]))),"common.R"))
assert_packages()
input <- read_teacher_csv(args[[1]])
outdir <- args[[2]]
dir.create(outdir,recursive=TRUE,showWarnings=FALSE)
for (field in c("Participant","WWR","Complexity","Gender","OrderGroup","ExperienceGroup")) input[[field]] <- factor(input[[field]])
outcomes <- c("TableShare","WindowShare","RawCompetition","LogTableEnrichment","LogWindowEnrichment","AdjustedCompetition")
rows <- list()
for (outcome in outcomes) {
  data <- input[is.finite(input[[outcome]]),,drop=FALSE]
  formula <- as.formula(paste0(outcome," ~ WWR * Complexity + WWR * ExperienceGroup + Complexity * ExperienceGroup + Gender + Block + PositionWithinBlockCentered + OrderGroup + (1|Participant)"))
  model <- fit_lmer_or_record(formula,data,outcome,"Sensitivity_EEGCommon")
  rows[[length(rows)+1]] <- if(inherits(model,"teacher_model_failure")) failure_row(model) else tidy_mixed(model,outcome,"Sensitivity_EEGCommon")
}
write_csv_utf8(bind_rows_fill(rows),file.path(outdir,"16_EEG_valid_common_sample_sensitivity.csv"))
write_session_info(outdir)
