args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) stop("usage: eye_stage2_analysis.R input.csv outdir threshold")
file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_dir <- dirname(normalizePath(sub("^--file=", "", file_arg[[1]])))
source(file.path(script_dir, "common.R"))
assert_packages()
input <- utils::read.csv(args[[1]], check.names = FALSE)
outdir <- args[[2]]
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
input$IncludedPrimary <- as.logical(input$IncludedPrimary)
input$EligibleNonTracking <- as.logical(input$EligibleNonTracking)
all_input <- input
input <- input[input$IncludedPrimary %in% TRUE, , drop = FALSE]

input$Participant <- factor(input$Participant)
input$WWR <- factor(input$WWR)
input$Complexity <- factor(input$Complexity)
input$Gender <- factor(input$Gender)
input$OrderGroup <- factor(input$OrderGroup)
input$ExperienceGroup <- factor(input$ExperienceGroup)

outcomes_a <- c("TableShare", "WindowShare", "RawCompetition")
outcomes_b <- c("LogTableEnrichment", "LogWindowEnrichment", "AdjustedCompetition")
formula_for <- function(outcome) {
  stats::as.formula(paste0(
    outcome,
    " ~ WWR * Complexity + WWR * ExperienceGroup + ",
    "Complexity * ExperienceGroup + Gender + Block + ",
    "PositionWithinBlockCentered + OrderGroup + (1|Participant)"
  ))
}

fixed_rows <- list()
cr2 <- list()
diagnostics <- list()
posthoc <- list()
for (outcome in c(outcomes_a, outcomes_b)) {
  data <- input[is.finite(input[[outcome]]), , drop = FALSE]
  is_share <- outcome %in% c("TableShare", "WindowShare")
  model <- if (is_share) {
    tryCatch(
      glmmTMB::glmmTMB(
        formula_for(outcome), data = data,
        family = glmmTMB::ordbeta(link = "logit")
      ),
      error = function(e) structure(
        list(
          outcome = outcome, model = "teacher_eye_ordered_beta_mixed",
          error = conditionMessage(e), formula = deparse(formula_for(outcome))
        ),
        class = "teacher_model_failure"
      )
    )
  } else {
    fit_lmer_or_record(formula_for(outcome), data, outcome, "teacher_eye_lmm")
  }
  if (inherits(model, "teacher_model_failure")) {
    diagnostics[[length(diagnostics) + 1]] <- failure_row(model)
    next
  }
  model_name <- if (is_share) "teacher_eye_ordered_beta_mixed" else "teacher_eye_lmm"
  fixed_rows[[length(fixed_rows) + 1]] <- tidy_mixed(
    model, outcome, model_name
  )
  # clubSandwich CR2 is evaluated with the pre-specified Gaussian
  # random-intercept companion for bounded shares; the ordered-beta fit
  # remains the primary likelihood model and no zero pseudoconstant is used.
  cr2_model <- if (is_share) {
    fit_lmer_or_record(
      formula_for(outcome), data, outcome, "teacher_eye_lmm_CR2_companion"
    )
  } else model
  if (!inherits(cr2_model, "teacher_model_failure")) {
    cr2[[length(cr2) + 1]] <- cr2_rows(
      cr2_model, outcome,
      if (is_share) "teacher_eye_lmm_CR2_companion" else model_name,
      data$Participant
    )
  }
  diagnostics[[length(diagnostics) + 1]] <- data.frame(
    outcome = outcome, model = model_name, status = "fit",
    singular = if (is_share) NA else lme4::isSingular(model),
    n = stats::nobs(model),
    stringsAsFactors = FALSE
  )
  emm <- emmeans::emmeans(model, ~ WWR)
  pair <- as.data.frame(emmeans::contrast(emm, "pairwise", adjust = "holm"))
  pair$outcome <- outcome
  posthoc[[length(posthoc) + 1]] <- pair
}

fixed <- bind_rows_fill(fixed_rows)
cr2_frame <- bind_rows_fill(cr2)
diag <- bind_rows_fill(diagnostics)
post <- bind_rows_fill(posthoc)
if (nrow(fixed)) {
  fixed$family <- ifelse(fixed$outcome %in% outcomes_a, "A", "B")
}
if (nrow(cr2_frame)) {
  cr2_frame$family <- ifelse(cr2_frame$outcome %in% outcomes_a, "A", "B")
  cr2_frame <- bh_within_family(cr2_frame, c("family", "term"))
  fixed <- merge(
    fixed,
    cr2_frame[, c(
      "outcome", "term", "std.error", "df", "p.value", "p.value.BH"
    )],
    by = c("outcome", "term"), all.x = TRUE,
    suffixes = c(".likelihood", ".CR2")
  )
}
write_csv_utf8(fixed[fixed$outcome %in% outcomes_a, , drop = FALSE],
               file.path(outdir, "09_familyA_primary_models.csv"))
write_csv_utf8(fixed[fixed$outcome %in% outcomes_b, , drop = FALSE],
               file.path(outdir, "10_familyB_primary_models.csv"))
write_csv_utf8(post, file.path(outdir, "11_WWR_posthoc_Holm.csv"))
write_csv_utf8(cr2_frame, file.path(outdir, "12_CR2_robust_results.csv"))
write_csv_utf8(diag, file.path(outdir, "17_core_model_diagnostics.csv"))

# Required, explicitly labelled sensitivity fits.
block1 <- input[input$Block == 1, , drop = FALSE]
write_csv_utf8(block1, file.path(outdir, "14_block1_sensitivity_input.csv"))
common <- input[as.logical(input$IncludeEEGValid), , drop = FALSE]
write_csv_utf8(common, file.path(outdir, "15_EEG_valid_common_sample_input.csv"))

sensitivity_rows <- list()
thresholds <- c(.50, .60, .70)
for (threshold in thresholds) {
  data_threshold <- all_input[
    all_input$EligibleNonTracking %in% TRUE &
      is.finite(all_input$ValidTrackingRatio) &
      all_input$ValidTrackingRatio >= threshold,
    , drop = FALSE
  ]
  for (outcome in c(outcomes_a, outcomes_b)) {
    data <- data_threshold[is.finite(data_threshold[[outcome]]), , drop = FALSE]
    model <- fit_lmer_or_record(
      formula_for(outcome), data, outcome,
      paste0("TrackingThreshold_", threshold)
    )
    row <- if (inherits(model, "teacher_model_failure")) {
      failure_row(model)
    } else tidy_mixed(model, outcome, paste0("TrackingThreshold_", threshold))
    row$Threshold <- threshold
    sensitivity_rows[[length(sensitivity_rows) + 1]] <- row
  }
}
for (label in c("Block1", "EEGCommon")) {
  subset <- if (label == "Block1") {
    input[input$Block == 1, , drop = FALSE]
  } else input[as.logical(input$IncludeEEGValid), , drop = FALSE]
  for (outcome in c(outcomes_a, outcomes_b)) {
    data <- subset[is.finite(subset[[outcome]]), , drop = FALSE]
    model <- fit_lmer_or_record(
      formula_for(outcome), data, outcome, paste0("Sensitivity_", label)
    )
    sensitivity_rows[[length(sensitivity_rows) + 1]] <-
      if (inherits(model, "teacher_model_failure")) failure_row(model) else
        tidy_mixed(model, outcome, paste0("Sensitivity_", label))
  }
}
for (excluded in levels(input$Participant)) {
  subset <- input[input$Participant != excluded, , drop = FALSE]
  for (outcome in c(outcomes_a, outcomes_b)) {
    data <- subset[is.finite(subset[[outcome]]), , drop = FALSE]
    model <- fit_lmer_or_record(
      formula_for(outcome), data, outcome, "Sensitivity_LOO"
    )
    row <- if (inherits(model, "teacher_model_failure")) {
      failure_row(model)
    } else tidy_mixed(model, outcome, "Sensitivity_LOO")
    row$ExcludedParticipant <- excluded
    sensitivity_rows[[length(sensitivity_rows) + 1]] <- row
  }
}
write_csv_utf8(
  bind_rows_fill(sensitivity_rows),
  file.path(outdir, "16_eye_structural_sensitivities.csv")
)
write_session_info(outdir)
