args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("usage: eeg_crossmodal_analysis.R input outdir metrics")
}
file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_dir <- dirname(normalizePath(sub("^--file=", "", file_arg[[1]])))
source(file.path(script_dir, "common.R"))
assert_packages()
input <- utils::read.csv(args[[1]], check.names = FALSE)
outdir <- args[[2]]
metrics <- strsplit(args[[3]], ",", fixed = TRUE)[[1]]
metrics <- metrics[nzchar(metrics) & metrics %in% names(input)]
if (!length(metrics) || length(metrics) > 2) {
  stop("Cross-modal analysis requires exactly 1-2 pre-registered EEG metrics.")
}
for (column in c(
  "Participant", "WWR", "Complexity", "ExperienceGroup", "Gender", "OrderGroup"
)) input[[column]] <- factor(input[[column]])

rows <- list()
cr2 <- list()
diagnostics <- list()
for (metric in metrics) {
  for (eye in c("RawCompetition", "AdjustedCompetition")) {
    subset <- input[
      is.finite(input[[metric]]) & is.finite(input[[eye]]),
      , drop = FALSE
    ]
    formula <- stats::as.formula(paste0(
      metric, " ~ ", eye,
      " + WWR * Complexity + ExperienceGroup + Gender + Block + ",
      "PositionWithinBlockCentered + OrderGroup + (1|Participant)"
    ))
    model <- fit_lmer_or_record(
      formula, subset, metric, paste0("crossmodal_", eye)
    )
    if (inherits(model, "teacher_model_failure")) {
      diagnostics[[length(diagnostics) + 1]] <- failure_row(model)
      next
    }
    row <- tidy_mixed(model, metric, paste0("crossmodal_", eye))
    row$EyePredictor <- eye
    rows[[length(rows) + 1]] <- row
    robust <- cr2_rows(
      model, metric, paste0("crossmodal_", eye), subset$Participant
    )
    robust$EyePredictor <- eye
    cr2[[length(cr2) + 1]] <- robust
    diagnostics[[length(diagnostics) + 1]] <- data.frame(
      outcome = metric, model = paste0("crossmodal_", eye),
      status = "fit", singular = lme4::isSingular(model),
      n = stats::nobs(model)
    )
  }
}
write_csv_utf8(
  bind_rows_fill(rows), file.path(outdir, "19_eeg_crossmodal_results.csv")
)
write_csv_utf8(
  bind_rows_fill(cr2), file.path(outdir, "19b_eeg_crossmodal_CR2.csv")
)
write_csv_utf8(
  bind_rows_fill(diagnostics),
  file.path(outdir, "19c_eeg_crossmodal_diagnostics.csv")
)
