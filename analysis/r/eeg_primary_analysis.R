args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 7) {
  stop(paste(
    "usage: eeg_primary_analysis.R input outdir core_relative",
    "core_absolute iterations secondary_relative supplemental_relative"
  ))
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
secondary <- strsplit(args[[6]], ",", fixed = TRUE)[[1]]
supplemental <- strsplit(args[[7]], ",", fixed = TRUE)[[1]]
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
model0_formula <- function(outcome) stats::as.formula(paste0(
  outcome,
  " ~ WWR * Complexity + WWR * ExperienceGroup + ",
  "Complexity * ExperienceGroup + Gender + (1|Participant)"
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
secondary_fit <- fit_family(secondary, "relative_secondary")
supplemental_fit <- fit_family(supplemental, "relative_supplemental")
model0_fixed <- list()
model0_posthoc <- list()
overall_tests <- list()
comparison_rows <- list()
for (outcome in relative[nzchar(relative)]) {
  data <- input[is.finite(input[[outcome]]), , drop = FALSE]
  model0 <- fit_lmer_or_record(
    model0_formula(outcome), data, outcome, "EEG_relative_model0"
  )
  model1 <- primary$models[[outcome]]
  if (inherits(model0, "teacher_model_failure")) {
    model0_fixed[[length(model0_fixed) + 1]] <- failure_row(model0)
    next
  }
  model0_tidy <- tidy_mixed(model0, outcome, "EEG_relative_model0")
  model0_fixed[[length(model0_fixed) + 1]] <- model0_tidy
  if (!is.null(model1)) {
    joint <- as.data.frame(emmeans::joint_tests(model1))
    joint$outcome <- outcome
    overall_tests[[length(overall_tests) + 1]] <- joint
    wwr_row <- joint[joint$`model term` %in% c("WWR", "WWR:Complexity"), ]
    if (nrow(wwr_row) && any(wwr_row$p.value < .05, na.rm = TRUE)) {
      pair <- as.data.frame(emmeans::contrast(
        emmeans::emmeans(model1, ~ WWR), "pairwise", adjust = "holm"
      ))
      pair$outcome <- outcome
      pair$PosthocGate <- "WWR_or_interaction_overall_p_below_0.05"
      model0_posthoc[[length(model0_posthoc) + 1]] <- pair
    }
    m1_tidy <- tidy_mixed(model1, outcome, "EEG_relative_model1")
    comparison <- merge(
      model0_tidy, m1_tidy,
      by = c("outcome", "term"), suffixes = c(".Model0", ".Model1")
    )
    if (nrow(comparison)) {
      comparison$EstimateChangePercent <- ifelse(
        abs(comparison$estimate.Model0) < 1e-12,
        NA_real_,
        100 * abs(comparison$estimate.Model1 - comparison$estimate.Model0) /
          abs(comparison$estimate.Model0)
      )
      comparison$DirectionConsistent <- (
        sign(comparison$estimate.Model0) == sign(comparison$estimate.Model1)
      )
      comparison_rows[[length(comparison_rows) + 1]] <- comparison
    }
  }
}
model0_fixed <- bind_rows_fill(model0_fixed)
gated_posthoc <- bind_rows_fill(model0_posthoc)
overall_tests <- bind_rows_fill(overall_tests)
model_comparison <- bind_rows_fill(comparison_rows)
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
adjust_nonprimary_family <- function(fit, family_label) {
  if (!nrow(fit$robust)) return(fit)
  fit$robust$correction_family <- paste(
    family_label,
    sub(".*_(theta|alpha|beta).*", "\\1", tolower(fit$robust$outcome)),
    fit$robust$term, sep = ":"
  )
  fit$robust <- bh_within_family(fit$robust, c("correction_family"))
  fit$fixed <- merge(
    fit$fixed,
    fit$robust[, c(
      "outcome", "term", "std.error", "df", "p.value",
      "p.value.BH", "correction_family"
    )],
    by = c("outcome", "term"), all.x = TRUE,
    suffixes = c(".likelihood", ".CR2")
  )
  fit
}
secondary_fit <- adjust_nonprimary_family(secondary_fit, "secondary")
supplemental_fit <- adjust_nonprimary_family(
  supplemental_fit, "supplemental"
)

sensitivity_rows <- list()
block1_rows <- list()
complete12_rows <- list()
previous_rows <- list()
loo_rows <- list()
gender_rows <- list()
extreme_rows <- list()
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
    row <- if (inherits(model, "teacher_model_failure")) failure_row(model) else
      tidy_mixed(model, outcome, paste0("Sensitivity_", label))
    sensitivity_rows[[length(sensitivity_rows) + 1]] <- row
    if (label == "Block1") block1_rows[[length(block1_rows) + 1]] <- row
    if (label == "Complete12") complete12_rows[[length(complete12_rows) + 1]] <- row
    if (label == "PreviousScene") previous_rows[[length(previous_rows) + 1]] <- row
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
    loo_rows[[length(loo_rows) + 1]] <- row
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
    gender_rows[[length(gender_rows) + 1]] <- row
  }
  limits <- stats::quantile(
    input[[outcome]], probs = c(.01, .99), na.rm = TRUE
  )
  trimmed <- input[
    input[[outcome]] >= limits[[1]] & input[[outcome]] <= limits[[2]],
    , drop = FALSE
  ]
  extreme_model <- fit_lmer_or_record(
    primary_formula(outcome), trimmed, outcome, "Sensitivity_ExtremeTrim1Percent"
  )
  extreme <- if (inherits(extreme_model, "teacher_model_failure")) {
    failure_row(extreme_model)
  } else {
    tidy_mixed(extreme_model, outcome, "Sensitivity_ExtremeTrim1Percent")
  }
  extreme$LowerCut <- limits[[1]]
  extreme$UpperCut <- limits[[2]]
  sensitivity_rows[[length(sensitivity_rows) + 1]] <- extreme
  extreme_rows[[length(extreme_rows) + 1]] <- extreme
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

write_csv_utf8(model0_fixed, file.path(outdir, "08_eeg_model0_results.csv"))
write_csv_utf8(primary$fixed, file.path(outdir, "09_eeg_primary_model1_results.csv"))
write_csv_utf8(overall_tests, file.path(outdir, "09b_eeg_model1_overall_tests.csv"))
write_csv_utf8(gated_posthoc, file.path(outdir, "10_eeg_posthoc.csv"))
write_csv_utf8(model_comparison, file.path(outdir, "11_eeg_model0_model1_comparison.csv"))
write_csv_utf8(primary$fixed, file.path(outdir, "04_eeg_primary_models_BH.csv"))
write_csv_utf8(
  secondary_fit$fixed, file.path(outdir, "09c_eeg_secondary_models.csv")
)
write_csv_utf8(
  supplemental_fit$fixed,
  file.path(outdir, "09d_eeg_supplemental_models.csv")
)
write_csv_utf8(
  secondary_fit$robust, file.path(outdir, "12b_eeg_secondary_CR2.csv")
)
write_csv_utf8(
  supplemental_fit$robust,
  file.path(outdir, "12c_eeg_supplemental_CR2.csv")
)
write_csv_utf8(gated_posthoc, file.path(outdir, "05_eeg_WWR_posthoc_Holm.csv"))
write_csv_utf8(primary$robust, file.path(outdir, "06_eeg_CR2_results.csv"))
write_csv_utf8(
  bind_rows_fill(boot_rows),
  file.path(outdir, "07_eeg_cluster_bootstrap_5000.csv")
)
write_csv_utf8(
  bind_rows_fill(boot_failures),
  file.path(outdir, "08_eeg_bootstrap_failures.csv")
)
write_csv_utf8(primary$robust, file.path(outdir, "12_eeg_cr2_results.csv"))
write_csv_utf8(
  bind_rows_fill(boot_rows),
  file.path(outdir, "13_eeg_bootstrap_results.csv")
)
write_csv_utf8(absolute_fit$fixed, file.path(outdir, "13_absolute_power_sensitivity.csv"))
write_csv_utf8(sensitivity, file.path(outdir, "12_eeg_structural_sensitivities.csv"))
write_csv_utf8(block1_rows |> bind_rows_fill(), file.path(outdir, "14_eeg_Block1_sensitivity.csv"))
write_csv_utf8(complete12_rows |> bind_rows_fill(), file.path(outdir, "14b_eeg_complete12_sensitivity.csv"))
write_csv_utf8(previous_rows |> bind_rows_fill(), file.path(outdir, "15_eeg_previous_condition_integration.csv"))
write_csv_utf8(loo_rows |> bind_rows_fill(), file.path(outdir, "16_eeg_leave_one_out.csv"))
write_csv_utf8(absolute_fit$fixed, file.path(outdir, "17_eeg_absolute_power_sensitivity.csv"))
write_csv_utf8(gender_rows |> bind_rows_fill(), file.path(outdir, "18_eeg_gender_sensitivity.csv"))
write_csv_utf8(extreme_rows |> bind_rows_fill(), file.path(outdir, "18b_eeg_extreme_value_sensitivity.csv"))
write_csv_utf8(
  bind_rows_fill(list(
    primary$diagnostic, absolute_fit$diagnostic,
    secondary_fit$diagnostic, supplemental_fit$diagnostic
  )),
  file.path(outdir, "18_eeg_model_diagnostics.csv")
)

boot_summary <- bind_rows_fill(boot_failures)
bootstrap_results <- bind_rows_fill(boot_rows)
condition_fixed <- primary$fixed[
  grepl("WWR|Complexity|ExperienceGroup", primary$fixed$term),
  , drop = FALSE
]
grade_rows <- list()
required_models <- c(
  "Sensitivity_Block1", "Sensitivity_Complete12",
  "Sensitivity_PreviousScene", "Sensitivity_LOO",
  "Sensitivity_GenderStratified", "Sensitivity_ExtremeTrim1Percent"
)
if (nrow(condition_fixed)) {
  for (i in seq_len(nrow(condition_fixed))) {
    row <- condition_fixed[i, , drop = FALSE]
    outcome <- row$outcome[[1]]
    term <- row$term[[1]]
    estimate <- row$estimate[[1]]
    q_value <- if ("p.value.BH" %in% names(row)) row$p.value.BH[[1]] else NA_real_
    sens <- sensitivity[
      sensitivity$outcome == outcome & sensitivity$term == term &
        is.finite(sensitivity$estimate),
      , drop = FALSE
    ]
    direction_concordance <- if (
      nrow(sens) && is.finite(estimate) && estimate != 0
    ) mean(sign(sens$estimate) == sign(estimate)) else NA_real_
    complete <- all(required_models %in% unique(sens$model))
    boot <- bootstrap_results[
      bootstrap_results$outcome == outcome &
        bootstrap_results$term == term,
      , drop = FALSE
    ]
    boot_available <- nrow(boot) > 0 && is.finite(boot$ci_low[[1]]) &&
      is.finite(boot$ci_high[[1]])
    boot_excludes_zero <- boot_available && (
      boot$ci_low[[1]] > 0 || boot$ci_high[[1]] < 0
    )
    boot_failure_ok <- boot_available &&
      boot$failed_replicates[[1]] <= .10 * iterations
    stable <- isTRUE(direction_concordance >= .80) && complete
    grade <- if (
      is.finite(q_value) && q_value < .05 && stable &&
        boot_excludes_zero && boot_failure_ok
    ) {
      "Robust"
    } else if (
      stable && boot_failure_ok &&
        isTRUE(is.finite(q_value) && q_value < .05)
    ) {
      "Partially robust"
    } else {
      "Exploratory"
    }
    grade_rows[[length(grade_rows) + 1]] <- data.frame(
      outcome = outcome,
      term = term,
      estimate = estimate,
      adjusted_p = q_value,
      sensitivity_estimate_count = nrow(sens),
      sensitivity_direction_concordance = direction_concordance,
      required_sensitivity_families_complete = complete,
      bootstrap_ci_low = if (boot_available) boot$ci_low[[1]] else NA_real_,
      bootstrap_ci_high = if (boot_available) boot$ci_high[[1]] else NA_real_,
      bootstrap_failed_replicates = if (
        boot_available
      ) boot$failed_replicates[[1]] else iterations,
      grade = grade,
      reason = paste(
        "Robust requires CR2/BH q<0.05, >=80% direction concordance",
        "across Block1, complete12, previous-scene, LOO, gender and",
        "extreme-value sensitivities, a bootstrap CI excluding zero,",
        "and <=10% failed bootstrap replicates."
      ),
      stringsAsFactors = FALSE
    )
  }
}
grade <- bind_rows_fill(grade_rows)
missing_outcomes <- setdiff(relative[nzchar(relative)], unique(grade$outcome))
if (length(missing_outcomes)) {
  grade <- bind_rows_fill(list(
    grade,
    data.frame(
      outcome = missing_outcomes, term = NA, estimate = NA,
      adjusted_p = NA, sensitivity_estimate_count = 0,
      sensitivity_direction_concordance = NA,
      required_sensitivity_families_complete = FALSE,
      bootstrap_ci_low = NA, bootstrap_ci_high = NA,
      bootstrap_failed_replicates = iterations,
      grade = "Unsupported",
      reason = "No estimable registered condition coefficient.",
      stringsAsFactors = FALSE
    )
  ))
}
write_csv_utf8(grade, file.path(outdir, "19_eeg_robustness_classification.csv"))
write_csv_utf8(grade, file.path(outdir, "20_eeg_evidence_classification.csv"))

# Six teacher-requested figures are produced from the exact model input/results
# with base R so the plotting backend remains R-only and reproducible.
first_outcome <- relative[nzchar(relative)][[1]]
safe_png <- function(path, code) {
  grDevices::png(path, width = 1800, height = 1200, res = 220)
  tryCatch(force(code), error = function(e) {
    graphics::plot.new()
    graphics::text(.5, .5, paste("Plot failed:", conditionMessage(e)))
  })
  grDevices::dev.off()
}
safe_png(file.path(outdir, "Figure_EEG_conditions.png"), {
  condition <- stats::aggregate(
    input[[first_outcome]],
    input[c("WWR", "Complexity")], mean, na.rm = TRUE
  )
  names(condition)[[3]] <- "Mean"
  stats::interaction.plot(
    condition$WWR, condition$Complexity, condition$Mean,
    type = "b", xlab = "WWR (categorical)", ylab = first_outcome,
    trace.label = "Complexity", main = "EEG condition means"
  )
})
safe_png(file.path(outdir, "Figure_EEG_effect_forest.png"), {
  forest <- primary$robust[
    primary$robust$term != "(Intercept)" &
      is.finite(primary$robust$estimate),
    , drop = FALSE
  ]
  forest <- utils::head(forest, 30)
  if (!nrow(forest)) stop("No finite CR2 coefficients")
  y <- rev(seq_len(nrow(forest)))
  graphics::plot(
    forest$estimate, y, xlim = range(
      c(forest$estimate - 1.96 * forest$std.error,
        forest$estimate + 1.96 * forest$std.error)
    ),
    yaxt = "n", ylab = "", xlab = "Estimate (95% CR2 interval)",
    main = "EEG primary effects"
  )
  graphics::segments(
    forest$estimate - 1.96 * forest$std.error, y,
    forest$estimate + 1.96 * forest$std.error, y
  )
  graphics::abline(v = 0, lty = 2)
  graphics::axis(
    2, at = y,
    labels = paste(forest$outcome, forest$term, sep = ": "), las = 2,
    cex.axis = .55
  )
})
safe_png(file.path(outdir, "FigureS_EEG_Block_Position.png"), {
  temporal <- stats::aggregate(
    input[[first_outcome]],
    input[c("Block", "PositionWithinBlock")], mean, na.rm = TRUE
  )
  names(temporal)[[3]] <- "Mean"
  stats::interaction.plot(
    temporal$PositionWithinBlock, temporal$Block, temporal$Mean,
    type = "b", xlab = "Position within block", ylab = first_outcome,
    trace.label = "Block", main = "Block and position"
  )
})
safe_png(file.path(outdir, "FigureS_EEG_OrderGroup.png"), {
  graphics::boxplot(
    input[[first_outcome]] ~ input$OrderGroup,
    xlab = "OrderGroup", ylab = first_outcome,
    main = "EEG by presentation-order group"
  )
})
safe_png(file.path(outdir, "FigureS_EEG_Block1_comparison.png"), {
  main_rows <- primary$fixed[
    primary$fixed$outcome == first_outcome &
      primary$fixed$term != "(Intercept)", , drop = FALSE
  ]
  block_rows <- bind_rows_fill(block1_rows)
  block_rows <- block_rows[
    block_rows$outcome == first_outcome &
      block_rows$term != "(Intercept)", , drop = FALSE
  ]
  common <- intersect(main_rows$term, block_rows$term)
  comparison <- merge(
    main_rows[main_rows$term %in% common, c("term", "estimate")],
    block_rows[block_rows$term %in% common, c("term", "estimate")],
    by = "term", suffixes = c(".Full", ".Block1")
  )
  graphics::plot(
    comparison$estimate.Full, comparison$estimate.Block1,
    xlab = "Full model estimate", ylab = "Block 1 estimate",
    main = "Block 1 versus full model"
  )
  graphics::abline(a = 0, b = 1, lty = 2)
})
safe_png(file.path(outdir, "FigureS_EEG_leave_one_out.png"), {
  loo <- bind_rows_fill(loo_rows)
  loo <- loo[
    loo$outcome == first_outcome & loo$term != "(Intercept)" &
      is.finite(loo$estimate),
    , drop = FALSE
  ]
  selected <- if (nrow(loo)) loo$term[[1]] else ""
  values <- loo$estimate[loo$term == selected]
  graphics::boxplot(
    values, ylab = "Leave-one-out estimate",
    main = paste("Leave-one-out:", selected)
  )
  graphics::abline(h = 0, lty = 2)
})
write_session_info(outdir)
