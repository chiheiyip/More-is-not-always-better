args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("usage: questionnaire_teacher_analysis.R input outdir")
}
file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_dir <- dirname(normalizePath(sub("^--file=", "", file_arg[[1]])))
source(file.path(script_dir, "common.R"))
assert_packages()
input <- utils::read.csv(args[[1]], check.names = FALSE)
outdir <- args[[2]]
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

names(input)[names(input) == "participant_id"] <- "Participant"
names(input)[names(input) == "scene_id"] <- "GlobalTrialOrder"
input$PositionWithinBlockCentered <- input$PositionWithinBlock - 3.5
for (column in c(
  "Participant", "WWR", "Complexity", "ExperienceGroup", "Gender", "OrderGroup"
)) input[[column]] <- factor(input[[column]])

formula_for <- function(outcome) stats::as.formula(paste0(
  outcome,
  " ~ WWR * Complexity + WWR * ExperienceGroup + ",
  "Complexity * ExperienceGroup + Gender + Block + ",
  "PositionWithinBlockCentered + OrderGroup + (1|Participant)"
))

descriptive_rows <- list()
fixed_rows <- list()
cr2_rows_all <- list()
posthoc_rows <- list()
diagnostic_rows <- list()
for (outcome in intersect(paste0("S", 1:5), names(input))) {
  descriptive <- stats::aggregate(
    input[[outcome]],
    input[c("WWR", "Complexity")],
    function(x) c(
      n = sum(is.finite(x)), mean = mean(x, na.rm = TRUE),
      sd = stats::sd(x, na.rm = TRUE)
    )
  )
  descriptive_rows[[length(descriptive_rows) + 1]] <- data.frame(
    outcome = outcome,
    WWR = descriptive$WWR,
    Complexity = descriptive$Complexity,
    n = descriptive$x[, "n"],
    mean = descriptive$x[, "mean"],
    sd = descriptive$x[, "sd"]
  )
  data <- input[is.finite(input[[outcome]]), , drop = FALSE]
  model <- fit_lmer_or_record(
    formula_for(outcome), data, outcome, "questionnaire_teacher_LMM"
  )
  if (inherits(model, "teacher_model_failure")) {
    diagnostic_rows[[length(diagnostic_rows) + 1]] <- failure_row(model)
    next
  }
  fixed_rows[[length(fixed_rows) + 1]] <- tidy_mixed(
    model, outcome, "questionnaire_teacher_LMM"
  )
  robust <- cr2_rows(
    model, outcome, "questionnaire_teacher_LMM", data$Participant
  )
  robust$correction_family <- outcome
  robust <- bh_within_family(robust, "correction_family")
  cr2_rows_all[[length(cr2_rows_all) + 1]] <- robust
  posthoc <- as.data.frame(emmeans::contrast(
    emmeans::emmeans(model, ~ WWR), "pairwise", adjust = "holm"
  ))
  posthoc$outcome <- outcome
  posthoc$adjustment <- "Holm within outcome"
  posthoc_rows[[length(posthoc_rows) + 1]] <- posthoc
  diagnostic_rows[[length(diagnostic_rows) + 1]] <- data.frame(
    outcome = outcome, model = "questionnaire_teacher_LMM",
    status = "fit", singular = lme4::isSingular(model),
    n = stats::nobs(model), participants = length(unique(data$Participant))
  )
}

write_csv_utf8(
  bind_rows_fill(descriptive_rows),
  file.path(outdir, "Questionnaire_42_descriptive_statistics.csv")
)
write_csv_utf8(
  bind_rows_fill(fixed_rows),
  file.path(outdir, "Questionnaire_42_LMM_results.csv")
)
write_csv_utf8(
  bind_rows_fill(cr2_rows_all),
  file.path(outdir, "Questionnaire_42_CR2_BH_results.csv")
)
write_csv_utf8(
  bind_rows_fill(posthoc_rows),
  file.path(outdir, "Questionnaire_42_WWR_posthoc_Holm.csv")
)
write_csv_utf8(
  bind_rows_fill(diagnostic_rows),
  file.path(outdir, "Questionnaire_42_model_diagnostics.csv")
)
write_session_info(outdir)
