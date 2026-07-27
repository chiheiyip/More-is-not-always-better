args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("usage: eeg_order_analysis.R input.csv outdir")
file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_dir <- dirname(normalizePath(sub("^--file=", "", file_arg[[1]])))
source(file.path(script_dir, "common.R"))
assert_packages()
input <- utils::read.csv(args[[1]], check.names = FALSE)
outdir <- args[[2]]
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
input$Participant <- factor(input$Participant)
input$WWR <- factor(input$WWR)
input$Complexity <- factor(input$Complexity)
input$OrderGroup <- factor(input$OrderGroup)

reserved <- c(
  "Participant", "OrderGroup", "Block", "PositionWithinBlock",
  "PositionWithinBlockCentered", "GlobalTrialOrder", "SceneID", "WWR",
  "Complexity", "PreviousWWR", "PreviousComplexity", "Gender",
  "ExperienceGroup", "IncludeEEGValid"
)
numeric_candidates <- names(input)[vapply(input, is.numeric, logical(1))]
outcomes <- setdiff(numeric_candidates, c(
  "Block", "PositionWithinBlock", "PositionWithinBlockCentered",
  "GlobalTrialOrder", "SceneID"
))
outcomes <- setdiff(outcomes, reserved)
if (!length(outcomes)) stop("No numeric EEG outcomes found.")

rows <- list()
cr2 <- list()
for (outcome in outcomes) {
  formulas <- list(
    Model0 = stats::as.formula(paste0(
      outcome, " ~ WWR * Complexity + (1|Participant)"
    )),
    Model1 = stats::as.formula(paste0(
      outcome, " ~ WWR * Complexity + Block + ",
      "PositionWithinBlockCentered + OrderGroup + (1|Participant)"
    )),
    PreviousScene = stats::as.formula(paste0(
      outcome, " ~ WWR * Complexity + Block + PositionWithinBlockCentered + ",
      "OrderGroup + PreviousWWR + PreviousComplexity + (1|Participant)"
    ))
  )
  for (name in names(formulas)) {
    data <- input
    if (name == "PreviousScene") data <- data[data$PositionWithinBlock > 1, ]
    model <- fit_lmer_or_record(formulas[[name]], data, outcome, name)
    if (inherits(model, "teacher_model_failure")) {
      rows[[length(rows) + 1]] <- failure_row(model)
      next
    }
    tidy <- tidy_mixed(model, outcome, name)
    tidy$AIC <- stats::AIC(model)
    rows[[length(rows) + 1]] <- tidy
    cr2[[length(cr2) + 1]] <- cr2_rows(model, outcome, name, data$Participant)
  }
  block1 <- input[input$Block == 1, ]
  model_b1 <- fit_lmer_or_record(
    stats::as.formula(paste0(
      outcome, " ~ WWR * Complexity + PositionWithinBlockCentered + ",
      "OrderGroup + (1|Participant)"
    )),
    block1, outcome, "Block1"
  )
  rows[[length(rows) + 1]] <- if (inherits(model_b1, "teacher_model_failure")) {
    failure_row(model_b1)
  } else tidy_mixed(model_b1, outcome, "Block1")
}
all_rows <- bind_rows_fill(rows)
all_cr2 <- bind_rows_fill(cr2)
write_csv_utf8(all_rows, file.path(outdir, "05_eeg_temporal_models.csv"))
m0 <- all_rows[all_rows$model == "Model0" & all_rows$status != "fit_failed", ]
m1 <- all_rows[all_rows$model == "Model1" & all_rows$status != "fit_failed", ]
comparison <- merge(
  m0, m1, by = c("outcome", "term"), suffixes = c(".Model0", ".Model1")
)
if (nrow(comparison)) {
  comparison$DirectionModel0 <- sign(comparison$estimate.Model0)
  comparison$DirectionModel1 <- sign(comparison$estimate.Model1)
  comparison$ChangePercent <- 100 * (
    comparison$estimate.Model1 - comparison$estimate.Model0
  ) / abs(comparison$estimate.Model0)
}
write_csv_utf8(comparison, file.path(outdir, "06_eeg_model0_model1_comparison.csv"))
write_csv_utf8(
  all_rows[all_rows$model == "PreviousScene", ],
  file.path(outdir, "07_eeg_previous_scene_model.csv")
)
write_csv_utf8(
  all_rows[all_rows$model == "Block1", ],
  file.path(outdir, "08_eeg_block1_sensitivity.csv")
)
write_csv_utf8(all_cr2, file.path(outdir, "09_eeg_order_CR2.csv"))
write_session_info(outdir)
