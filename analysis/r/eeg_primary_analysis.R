args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 5) {
  stop("usage: eeg_primary_analysis.R input outdir relative absolute iterations")
}
file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_dir <- dirname(normalizePath(sub("^--file=", "", file_arg[[1]])))
source(file.path(script_dir, "common.R"))
assert_packages()
input <- utils::read.csv(args[[1]], check.names = FALSE)
outdir <- args[[2]]
relative <- strsplit(args[[3]], ",", fixed = TRUE)[[1]]
absolute <- strsplit(args[[4]], ",", fixed = TRUE)[[1]]
iterations <- as.integer(args[[5]])
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

for (column in c(
  "Participant", "WWR", "Complexity", "ExperienceGroup", "Gender", "OrderGroup"
)) input[[column]] <- factor(input[[column]])

primary_formula <- function(outcome) stats::as.formula(paste0(
  outcome,
  " ~ WWR * Complexity + WWR * ExperienceGroup + ",
  "Complexity * ExperienceGroup + Gender + Block + ",
  "PositionWithinBlockCentered + OrderGroup + (1|Participant)"
))

fit_family <- function(outcomes, label) {
  fixed <- list()
  robust <- list()
  posthoc <- list()
  diagnostic <- list()
  models <- list()
  for (outcome in outcomes[nzchar(outcomes)]) {
    data <- input[is.finite(input[[outcome]]), ]
    model <- fit_lmer_or_record(
      primary_formula(outcome), data, outcome, paste0("EEG_", label)
    )
    if (inherits(model, "teacher_model_failure")) {
      diagnostic[[length(diagnostic) + 1]] <- failure_row(model)
      next
    }
    models[[outcome]] <- model
    fixed[[length(fixed) + 1]] <- tidy_mixed(model, outcome, paste0("EEG_", label))
    robust[[length(robust) + 1]] <- cr2_rows(
      model, outcome, paste0("EEG_", label), data$Participant
    )
    pair <- as.data.frame(emmeans::contrast(
      emmeans::emmeans(model, ~ WWR), "pairwise", adjust = "holm"
    ))
    pair$outcome <- outcome
    posthoc[[length(posthoc) + 1]] <- pair
    diagnostic[[length(diagnostic) + 1]] <- data.frame(
      outcome = outcome, model = paste0("EEG_", label), status = "fit",
      singular = lme4::isSingular(model), n = stats::nobs(model)
    )
  }
  list(
    fixed = bind_rows_fill(fixed),
    robust = bind_rows_fill(robust),
    posthoc = bind_rows_fill(posthoc),
    diagnostic = bind_rows_fill(diagnostic),
    models = models
  )
}

primary <- fit_family(relative, "relative_primary")
absolute_fit <- fit_family(absolute, "absolute_sensitivity")
if (nrow(primary$robust)) {
  primary$robust$correction_family <- paste(
    sub(".*_(theta|alpha|beta).*", "\\1", tolower(primary$robust$outcome)),
    primary$robust$term, sep = ":"
  )
  primary$robust <- bh_within_family(
    primary$robust, c("correction_family")
  )
  primary$fixed <- merge(
    primary$fixed,
    primary$robust[, c(
      "outcome", "term", "std.error", "df", "p.value",
      "p.value.BH", "correction_family"
    )],
    by = c("outcome", "term"), all.x = TRUE,
    suffixes = c(".likelihood", ".CR2")
  )
}

sensitivity_rows <- list()
for (outcome in relative[nzchar(relative)]) {
  sensitivity_sets <- list(
    Block1 = input[input$Block == 1, , drop = FALSE],
    Complete12 = input[
      input$Participant %in% names(which(table(input$Participant) == 12)),
      , drop = FALSE
    ],
    PreviousScene = input[input$PositionWithinBlock > 1, , drop = FALSE]
  )
  for (label in names(sensitivity_sets)) {
    data <- sensitivity_sets[[label]]
    formula <- if (label == "PreviousScene") {
      stats::as.formula(paste0(
        outcome,
        " ~ WWR * Complexity + WWR * ExperienceGroup + ",
        "Complexity * ExperienceGroup + PreviousWWR + PreviousComplexity + ",
        "Gender + Block + PositionWithinBlockCentered + OrderGroup + ",
        "(1|Participant)"
      ))
    } else primary_formula(outcome)
    model <- fit_lmer_or_record(formula, data, outcome, paste0("Sensitivity_", label))
    sensitivity_rows[[length(sensitivity_rows) + 1]] <-
      if (inherits(model, "teacher_model_failure")) failure_row(model) else
        tidy_mixed(model, outcome, paste0("Sensitivity_", label))
  }
  # Leave-one-participant-out estimates are kept as estimates, not silently
  # replaced when a refit fails.
  for (excluded in levels(input$Participant)) {
    data <- input[input$Participant != excluded, , drop = FALSE]
    model <- fit_lmer_or_record(
      primary_formula(outcome), data, outcome, "Sensitivity_LOO"
    )
    row <- if (inherits(model, "teacher_model_failure")) {
      failure_row(model)
    } else {
      tidy_mixed(model, outcome, "Sensitivity_LOO")
    }
    row$ExcludedParticipant <- excluded
    sensitivity_rows[[length(sensitivity_rows) + 1]] <- row
  }
  for (gender in levels(input$Gender)) {
    data <- input[input$Gender == gender, , drop = FALSE]
    # Gender is constant within a stratified sensitivity and is removed.
    formula <- stats::as.formula(paste0(
      outcome,
      " ~ WWR * Complexity + WWR * ExperienceGroup + ",
      "Complexity * ExperienceGroup + Block + ",
      "PositionWithinBlockCentered + OrderGroup + (1|Participant)"
    ))
    model <- fit_lmer_or_record(
      formula, data, outcome, "Sensitivity_GenderStratified"
    )
    row <- if (inherits(model, "teacher_model_failure")) {
      failure_row(model)
    } else tidy_mixed(model, outcome, "Sensitivity_GenderStratified")
    row$GenderStratum <- gender
    sensitivity_rows[[length(sensitivity_rows) + 1]] <- row
  }
}
sensitivity <- bind_rows_fill(sensitivity_rows)

# Cluster bootstrap: preserve failed replicates and never substitute another model.
set.seed(20260726)
boot_rows <- list()
boot_failures <- list()
participants <- levels(input$Participant)
for (outcome in names(primary$models)) {
  original_terms <- names(lme4::fixef(primary$models[[outcome]]))
  estimates <- matrix(
    NA_real_, nrow = iterations, ncol = length(original_terms),
    dimnames = list(NULL, original_terms)
  )
  failed <- character()
  for (b in seq_len(iterations)) {
    sampled <- sample(participants, length(participants), replace = TRUE)
    pieces <- lapply(seq_along(sampled), function(i) {
      part <- input[input$Participant == sampled[[i]], , drop = FALSE]
      part$Participant <- factor(paste0("boot_", i))
      part
    })
    boot_data <- do.call(rbind, pieces)
    model <- fit_lmer_or_record(
      primary_formula(outcome), boot_data, outcome, "cluster_bootstrap"
    )
    if (inherits(model, "teacher_model_failure")) {
      failed <- c(failed, conditionMessage(simpleError(model$error)))
      next
    }
    coef <- lme4::fixef(model)
    estimates[b, intersect(names(coef), original_terms)] <-
      coef[intersect(names(coef), original_terms)]
  }
  boot_rows[[length(boot_rows) + 1]] <- data.frame(
    outcome = outcome, term = original_terms,
    bootstrap_median = apply(estimates, 2, stats::median, na.rm = TRUE),
    ci_low = apply(estimates, 2, stats::quantile, probs = .025, na.rm = TRUE),
    ci_high = apply(estimates, 2, stats::quantile, probs = .975, na.rm = TRUE),
    successful = colSums(is.finite(estimates)),
    failed_replicates = length(failed)
  )
  boot_failures[[length(boot_failures) + 1]] <- data.frame(
    outcome = outcome, requested = iterations,
    failed_replicates = length(failed),
    failure_examples = paste(utils::head(unique(failed), 5), collapse = " | ")
  )
}

write_csv_utf8(primary$fixed, file.path(outdir, "04_eeg_primary_models_BH.csv"))
write_csv_utf8(primary$posthoc, file.path(outdir, "05_eeg_WWR_posthoc_Holm.csv"))
write_csv_utf8(primary$robust, file.path(outdir, "06_eeg_CR2_results.csv"))
write_csv_utf8(
  bind_rows_fill(boot_rows),
  file.path(outdir, "07_eeg_cluster_bootstrap_5000.csv")
)
write_csv_utf8(
  bind_rows_fill(boot_failures),
  file.path(outdir, "08_eeg_bootstrap_failures.csv")
)
write_csv_utf8(absolute_fit$fixed, file.path(outdir, "13_absolute_power_sensitivity.csv"))
write_csv_utf8(sensitivity, file.path(outdir, "12_eeg_structural_sensitivities.csv"))
write_csv_utf8(
  bind_rows_fill(list(primary$diagnostic, absolute_fit$diagnostic)),
  file.path(outdir, "18_eeg_model_diagnostics.csv")
)

boot_summary <- bind_rows_fill(boot_failures)
grade <- data.frame(
  outcome = relative[nzchar(relative)],
  grade = "Partially robust",
  reason = "Primary, CR2, bootstrap, and structural sensitivities require joint review.",
  stringsAsFactors = FALSE
)
if (nrow(grade)) {
  fitted <- unique(primary$fixed$outcome)
  grade$grade[!grade$outcome %in% fitted] <- "Unsupported"
  if (nrow(boot_summary)) {
    ok <- boot_summary$failed_replicates <= 0.1 * boot_summary$requested
    robust_candidates <- boot_summary$outcome[ok]
    grade$grade[
      grade$outcome %in% robust_candidates & grade$outcome %in% fitted
    ] <- "Robust"
  }
}
write_csv_utf8(grade, file.path(outdir, "19_eeg_robustness_classification.csv"))
write_session_info(outdir)
