args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) stop("usage: eeg_order_analysis.R input.csv outdir outcomes")
file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_dir <- dirname(normalizePath(sub("^--file=", "", file_arg[[1]])))
source(file.path(script_dir, "common.R"))
assert_packages()
input <- utils::read.csv(args[[1]], check.names = FALSE)
outdir <- args[[2]]
outcomes <- strsplit(args[[3]], ",", fixed = TRUE)[[1]]
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
input$Participant <- factor(input$Participant)
input$WWR <- factor(input$WWR)
input$Complexity <- factor(input$Complexity)
input$OrderGroup <- factor(input$OrderGroup)
input$ExperienceGroup <- factor(input$ExperienceGroup)
input$Gender <- factor(input$Gender)
outcomes <- outcomes[nzchar(outcomes) & outcomes %in% names(input)]
if (!length(outcomes)) stop("No numeric EEG outcomes found.")

rows <- list()
cr2 <- list()
for (outcome in outcomes) {
  formulas <- list(
    Model0 = stats::as.formula(paste0(
      outcome, " ~ WWR * Complexity + WWR * ExperienceGroup + ",
      "Complexity * ExperienceGroup + Gender + (1|Participant)"
    )),
    Model1 = stats::as.formula(paste0(
      outcome, " ~ WWR * Complexity + WWR * ExperienceGroup + ",
      "Complexity * ExperienceGroup + Gender + Block + ",
      "PositionWithinBlockCentered + OrderGroup + (1|Participant)"
    )),
    PreviousScene = stats::as.formula(paste0(
      outcome, " ~ WWR * Complexity + WWR * ExperienceGroup + ",
      "Complexity * ExperienceGroup + Gender + Block + ",
      "PositionWithinBlockCentered + OrderGroup + PreviousWWR + ",
      "PreviousComplexity + (1|Participant)"
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
      "WWR * ExperienceGroup + Complexity * ExperienceGroup + Gender + ",
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
write_csv_utf8(
  all_rows[all_rows$model == "Model1", ],
  file.path(outdir, "eeg_temporal_main_model.csv")
)
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
  comparison, file.path(outdir, "eeg_model0_model1_comparison.csv")
)
write_csv_utf8(
  all_rows[all_rows$model == "PreviousScene", ],
  file.path(outdir, "07_eeg_previous_scene_model.csv")
)
write_csv_utf8(
  all_rows[all_rows$model == "PreviousScene", ],
  file.path(outdir, "eeg_previous_scene_model.csv")
)
write_csv_utf8(
  all_rows[all_rows$model == "Block1", ],
  file.path(outdir, "08_eeg_block1_sensitivity.csv")
)
write_csv_utf8(
  all_rows[all_rows$model == "Block1", ],
  file.path(outdir, "eeg_round1_sensitivity.csv")
)
write_csv_utf8(all_cr2, file.path(outdir, "09_eeg_order_CR2.csv"))
write_csv_utf8(all_cr2, file.path(outdir, "eeg_cr2_robust_results.csv"))

figure_dir <- file.path(outdir, "eeg_temporal_figures")
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)
for (outcome in outcomes) {
  grDevices::png(
    file.path(figure_dir, paste0(outcome, "_Block_Position.png")),
    width = 1800, height = 1200, res = 220
  )
  temporal <- stats::aggregate(
    input[[outcome]],
    input[c("Block", "PositionWithinBlock")], mean, na.rm = TRUE
  )
  names(temporal)[[3]] <- "Mean"
  stats::interaction.plot(
    temporal$PositionWithinBlock, temporal$Block, temporal$Mean,
    type = "b", xlab = "Position within block", ylab = outcome,
    trace.label = "Block",
    main = paste("Descriptive temporal trend:", outcome)
  )
  grDevices::dev.off()
}
write_session_info(outdir)
