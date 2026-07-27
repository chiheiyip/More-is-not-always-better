args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) stop("usage: eye_stage3_analysis.R input outdir triggers")
file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_dir <- dirname(normalizePath(sub("^--file=", "", file_arg[[1]])))
source(file.path(script_dir, "common.R"))
assert_packages()
input <- utils::read.csv(args[[1]], check.names = FALSE)
outdir <- args[[2]]
triggers <- strsplit(toupper(args[[3]]), ",", fixed = TRUE)[[1]]
triggers <- triggers[nzchar(triggers)]
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

for (trigger in triggers) {
  if (trigger == "A") {
    zero <- data.frame(
      Outcome = c("TableShare", "WindowShare"),
      ZeroRate = c(mean(input$TableShare == 0, na.rm = TRUE),
                   mean(input$WindowShare == 0, na.rm = TRUE))
    )
    write_csv_utf8(zero, file.path(outdir, "02_zero_value_diagnostics.csv"))
  } else if (trigger == "D" || trigger == "K") {
    compare <- data.frame(
      Metric = c("RawCompetition", "AdjustedCompetition"),
      Mean = c(mean(input$RawCompetition, na.rm = TRUE),
               mean(input$AdjustedCompetition, na.rm = TRUE))
    )
    write_csv_utf8(compare, file.path(outdir, "05_area_interpretation_check.csv"))
  } else if (trigger == "E" || trigger == "F") {
    carry <- input[, intersect(
      c("Participant", "Block", "PositionWithinBlock", "PreviousWWR",
        "PreviousComplexity", "RawCompetition", "AdjustedCompetition"),
      names(input)
    ), drop = FALSE]
    write_csv_utf8(carry, file.path(outdir, "06_time_carryover_input.csv"))
  } else if (trigger == "H") {
    equipment <- input[input$Complexity == "C1", , drop = FALSE]
    write_csv_utf8(equipment, file.path(outdir, "09_equipment_C1_input.csv"))
  } else if (trigger == "J") {
    known <- c(
      "Participant", "GlobalTrialOrder", "Block", "PositionWithinBlock",
      "TableShare", "WindowShare", "RawCompetition", "AdjustedCompetition",
      "IncludedPrimary", "AOIImageID", "WWR", "Complexity"
    )
    eeg <- setdiff(names(input), known)
    eeg <- eeg[vapply(input[eeg], is.numeric, logical(1))]
    rows <- list()
    for (metric in eeg) {
      for (eye in c("RawCompetition", "AdjustedCompetition")) {
        complete <- is.finite(input[[metric]]) & is.finite(input[[eye]])
        rows[[length(rows) + 1]] <- data.frame(
          EEGMetric = metric, EyeMetric = eye,
          Participants = length(unique(input$Participant[complete])),
          Trials = sum(complete),
          Correlation = stats::cor(
            input[[metric]][complete], input[[eye]][complete]
          )
        )
      }
    }
    write_csv_utf8(
      bind_rows_fill(rows),
      file.path(outdir, "15_S3_cross_modal_prespecified_results.csv")
    )
  } else {
    note <- data.frame(
      Trigger = trigger,
      Status = "approved_but_requires_prespecified_external_or_manual_input"
    )
    write_csv_utf8(note, file.path(outdir, paste0("trigger_", trigger, "_status.csv")))
  }
}
write_session_info(outdir)
