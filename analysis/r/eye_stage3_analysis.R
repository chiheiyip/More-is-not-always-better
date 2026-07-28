args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 6) {
  stop("usage: eye_stage3_analysis.R input outdir triggers aoi boundary iterations")
}
file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_dir <- dirname(normalizePath(sub("^--file=", "", file_arg[[1]])))
source(file.path(script_dir, "common.R"))
assert_packages()
input <- utils::read.csv(args[[1]], check.names = FALSE)
outdir <- args[[2]]
triggers <- strsplit(toupper(args[[3]]), ",", fixed = TRUE)[[1]]
triggers <- triggers[nzchar(triggers)]
aoi_path <- args[[4]]
boundary_path <- args[[5]]
iterations <- as.integer(args[[6]])
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

input <- input[input$IncludedPrimary %in% TRUE, , drop = FALSE]
for (column in c(
  "Participant", "WWR", "Complexity", "ExperienceGroup", "Gender", "OrderGroup",
  "PreviousWWR", "PreviousComplexity"
)) {
  if (column %in% names(input)) input[[column]] <- factor(input[[column]])
}

primary_formula <- function(outcome, extra = "") {
  stats::as.formula(paste0(
    outcome,
    " ~ WWR * Complexity + WWR * ExperienceGroup + ",
    "Complexity * ExperienceGroup + Gender + Block + ",
    "PositionWithinBlockCentered + OrderGroup", extra, " + (1|Participant)"
  ))
}

fit_lmm_table <- function(data, outcomes, label, extra = "") {
  fixed <- list()
  robust <- list()
  diagnostic <- list()
  for (outcome in outcomes) {
    if (!outcome %in% names(data)) next
    subset <- data[is.finite(data[[outcome]]), , drop = FALSE]
    model <- fit_lmer_or_record(
      primary_formula(outcome, extra), subset, outcome, label
    )
    if (inherits(model, "teacher_model_failure")) {
      diagnostic[[length(diagnostic) + 1]] <- failure_row(model)
      next
    }
    fixed[[length(fixed) + 1]] <- tidy_mixed(model, outcome, label)
    robust[[length(robust) + 1]] <- cr2_rows(
      model, outcome, label, subset$Participant
    )
    diagnostic[[length(diagnostic) + 1]] <- data.frame(
      outcome = outcome, model = label, status = "fit",
      singular = lme4::isSingular(model), n = stats::nobs(model)
    )
  }
  list(
    fixed = bind_rows_fill(fixed),
    robust = bind_rows_fill(robust),
    diagnostics = bind_rows_fill(diagnostic)
  )
}

fit_aoi_outcome <- function(data, outcome, label) {
  subset <- data[is.finite(data[[outcome]]), , drop = FALSE]
  formula <- primary_formula(outcome)
  if (outcome == "Visited") {
    model <- tryCatch(
      glmmTMB::glmmTMB(formula, data = subset, family = stats::binomial()),
      error = function(e) structure(
        list(
          outcome = outcome, model = paste0(label, "_binomial_GLMM"),
          error = conditionMessage(e), formula = deparse(formula)
        ),
        class = "teacher_model_failure"
      )
    )
    model_name <- paste0(label, "_binomial_GLMM")
  } else if (outcome == "AttentionShare") {
    model <- tryCatch(
      glmmTMB::glmmTMB(
        formula, data = subset, family = glmmTMB::ordbeta(link = "logit")
      ),
      error = function(e) structure(
        list(
          outcome = outcome, model = paste0(label, "_ordered_beta"),
          error = conditionMessage(e), formula = deparse(formula)
        ),
        class = "teacher_model_failure"
      )
    )
    model_name <- paste0(label, "_ordered_beta")
  } else {
    model <- fit_lmer_or_record(formula, subset, outcome, paste0(label, "_LMM"))
    model_name <- paste0(label, "_LMM")
  }
  if (inherits(model, "teacher_model_failure")) {
    return(list(fixed = failure_row(model), diagnostic = failure_row(model)))
  }
  list(
    fixed = tidy_mixed(model, outcome, model_name),
    diagnostic = data.frame(
      outcome = outcome, model = model_name, status = "fit",
      singular = if (inherits(model, "merMod")) lme4::isSingular(model) else NA,
      n = stats::nobs(model)
    )
  )
}

all_diagnostics <- list()

if ("A" %in% triggers) {
  zero <- data.frame(
    Outcome = c("TableShare", "WindowShare"),
    ZeroRate = c(
      mean(input$TableShare == 0, na.rm = TRUE),
      mean(input$WindowShare == 0, na.rm = TRUE)
    )
  )
  write_csv_utf8(zero, file.path(outdir, "02_zero_value_diagnostics.csv"))
  two_part <- list()
  for (outcome in c("TableShare", "WindowShare")) {
    visited_name <- paste0(outcome, "_visited")
    input[[visited_name]] <- as.integer(input[[outcome]] > 0)
    visit_formula <- primary_formula(visited_name)
    visit_model <- tryCatch(
      glmmTMB::glmmTMB(
        visit_formula, data = input, family = stats::binomial()
      ),
      error = function(e) structure(
        list(
          outcome = visited_name, model = "two_part_visit_GLMM",
          error = conditionMessage(e), formula = deparse(visit_formula)
        ),
        class = "teacher_model_failure"
      )
    )
    if (inherits(visit_model, "teacher_model_failure")) {
      two_part[[length(two_part) + 1]] <- failure_row(visit_model)
    } else {
      two_part[[length(two_part) + 1]] <- tidy_mixed(
        visit_model, visited_name, "two_part_visit_GLMM"
      )
    }
    positive <- input[
      is.finite(input[[outcome]]) & input[[outcome]] > 0,
      , drop = FALSE
    ]
    positive_formula <- primary_formula(outcome)
    positive_model <- tryCatch(
      glmmTMB::glmmTMB(
        positive_formula, data = positive,
        family = glmmTMB::ordbeta(link = "logit")
      ),
      error = function(e) structure(
        list(
          outcome = outcome, model = "two_part_positive_ordered_beta",
          error = conditionMessage(e), formula = deparse(positive_formula)
        ),
        class = "teacher_model_failure"
      )
    )
    if (inherits(positive_model, "teacher_model_failure")) {
      two_part[[length(two_part) + 1]] <- failure_row(positive_model)
    } else {
      two_part[[length(two_part) + 1]] <- tidy_mixed(
        positive_model, outcome, "two_part_positive_ordered_beta"
      )
    }
  }
  write_csv_utf8(
    bind_rows_fill(two_part),
    file.path(outdir, "02b_zero_two_part_models.csv")
  )
}

if ("B" %in% triggers) {
  diagnostics_fit <- fit_lmm_table(
    input,
    c("RawCompetition", "AdjustedCompetition"),
    "stage3_diagnostic_alternative"
  )
  write_csv_utf8(
    diagnostics_fit$fixed,
    file.path(outdir, "04_diagnostic_alternative_models.csv")
  )
  write_csv_utf8(
    diagnostics_fit$robust,
    file.path(outdir, "04b_diagnostic_alternative_CR2.csv")
  )
  all_diagnostics[[length(all_diagnostics) + 1]] <-
    diagnostics_fit$diagnostics
}

if ("C" %in% triggers && nzchar(boundary_path) && file.exists(boundary_path)) {
  boundary <- utils::read.csv(boundary_path, check.names = FALSE)
  for (column in c(
    "Participant", "WWR", "Complexity", "ExperienceGroup", "Gender", "OrderGroup",
    "PreviousWWR", "PreviousComplexity", "BoundaryMarginPx"
  )) boundary[[column]] <- factor(boundary[[column]])
  boundary_rows <- list()
  for (margin in levels(boundary$BoundaryMarginPx)) {
    subset <- boundary[boundary$BoundaryMarginPx == margin, , drop = FALSE]
    fitted <- fit_lmm_table(
      subset,
      c("RawCompetition", "AdjustedCompetition"),
      paste0("AOI_boundary_", margin, "px")
    )
    boundary_rows[[length(boundary_rows) + 1]] <- fitted$fixed
    all_diagnostics[[length(all_diagnostics) + 1]] <- fitted$diagnostics
  }
  write_csv_utf8(
    bind_rows_fill(boundary_rows),
    file.path(outdir, "03b_AOI_boundary_sensitivity_models.csv")
  )
}

if ("D" %in% triggers || "K" %in% triggers) {
  compare <- data.frame(
    Metric = c("RawCompetition", "AdjustedCompetition"),
    Mean = c(
      mean(input$RawCompetition, na.rm = TRUE),
      mean(input$AdjustedCompetition, na.rm = TRUE)
    ),
    Median = c(
      stats::median(input$RawCompetition, na.rm = TRUE),
      stats::median(input$AdjustedCompetition, na.rm = TRUE)
    )
  )
  write_csv_utf8(
    compare, file.path(outdir, "05_area_interpretation_check.csv")
  )
  composition <- fit_lmm_table(
    input,
    c(
      "TableShare", "WindowShare", "RawCompetition",
      "LogTableEnrichment", "LogWindowEnrichment", "AdjustedCompetition"
    ),
    "area_and_composition_sensitivity"
  )
  write_csv_utf8(
    composition$fixed,
    file.path(outdir, "05b_area_composition_models.csv")
  )
  all_diagnostics[[length(all_diagnostics) + 1]] <-
    composition$diagnostics
}

if ("E" %in% triggers) {
  time_rows <- list()
  for (outcome in c("RawCompetition", "AdjustedCompetition")) {
    formula <- stats::as.formula(paste0(
      outcome,
      " ~ WWR * Complexity + ExperienceGroup + Gender + Block + ",
      "PositionWithinBlockCentered + I(PositionWithinBlockCentered^2) + ",
      "OrderGroup + (1|Participant)"
    ))
    model <- fit_lmer_or_record(formula, input, outcome, "time_linear_quadratic")
    time_rows[[length(time_rows) + 1]] <- if (
      inherits(model, "teacher_model_failure")
    ) failure_row(model) else tidy_mixed(model, outcome, "time_linear_quadratic")
  }
  write_csv_utf8(
    bind_rows_fill(time_rows),
    file.path(outdir, "06_time_effect_models.csv")
  )
}

if ("F" %in% triggers) {
  carry_input <- input[input$PositionWithinBlock > 1, , drop = FALSE]
  carry <- fit_lmm_table(
    carry_input,
    c("RawCompetition", "AdjustedCompetition"),
    "same_block_carryover",
    " + PreviousWWR + PreviousComplexity"
  )
  write_csv_utf8(
    carry$fixed, file.path(outdir, "07_carryover_models.csv")
  )
  write_csv_utf8(
    carry$robust, file.path(outdir, "07b_carryover_CR2.csv")
  )
  all_diagnostics[[length(all_diagnostics) + 1]] <- carry$diagnostics
}

if ("G" %in% triggers) {
  experience <- fit_lmm_table(
    input,
    c("RawCompetition", "AdjustedCompetition"),
    "experience_moderation"
  )
  write_csv_utf8(
    experience$fixed,
    file.path(outdir, "08_experience_moderation_models.csv")
  )
  write_csv_utf8(
    experience$robust,
    file.path(outdir, "08b_experience_moderation_CR2.csv")
  )
  simple_rows <- list()
  gender_rows <- list()
  boot_rows <- list()
  failure_rows <- list()
  set.seed(20260728)
  participants <- levels(input$Participant)
  for (outcome in c("RawCompetition", "AdjustedCompetition")) {
    model <- fit_lmer_or_record(
      primary_formula(outcome), input, outcome, "experience_moderation"
    )
    if (!inherits(model, "teacher_model_failure")) {
      simple <- as.data.frame(emmeans::emmeans(
        model, pairwise ~ WWR | ExperienceGroup, adjust = "holm"
      )$contrasts)
      simple$outcome <- outcome
      simple_rows[[length(simple_rows) + 1]] <- simple
      terms <- names(lme4::fixef(model))
      estimates <- matrix(
        NA_real_, nrow = iterations, ncol = length(terms),
        dimnames = list(NULL, terms)
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
        boot_model <- fit_lmer_or_record(
          primary_formula(outcome), boot_data, outcome,
          "experience_cluster_bootstrap"
        )
        if (inherits(boot_model, "teacher_model_failure")) {
          failed <- c(failed, boot_model$error)
          next
        }
        coefficient <- lme4::fixef(boot_model)
        common <- intersect(names(coefficient), terms)
        estimates[b, common] <- coefficient[common]
      }
      boot_rows[[length(boot_rows) + 1]] <- data.frame(
        outcome = outcome,
        term = terms,
        bootstrap_median = apply(
          estimates, 2, stats::median, na.rm = TRUE
        ),
        ci_low = apply(
          estimates, 2, stats::quantile, probs = .025, na.rm = TRUE
        ),
        ci_high = apply(
          estimates, 2, stats::quantile, probs = .975, na.rm = TRUE
        ),
        successful = colSums(is.finite(estimates)),
        failed_replicates = length(failed)
      )
      failure_rows[[length(failure_rows) + 1]] <- data.frame(
        outcome = outcome, requested = iterations,
        failed_replicates = length(failed),
        failure_examples = paste(
          utils::head(unique(failed), 5), collapse = " | "
        )
      )
    }
    for (gender in levels(input$Gender)) {
      subset <- input[input$Gender == gender, , drop = FALSE]
      formula <- stats::as.formula(paste0(
        outcome,
        " ~ WWR * Complexity + WWR * ExperienceGroup + ",
        "Complexity * ExperienceGroup + Block + ",
        "PositionWithinBlockCentered + OrderGroup + (1|Participant)"
      ))
      gender_model <- fit_lmer_or_record(
        formula, subset, outcome, "experience_gender_sensitivity"
      )
      row <- if (inherits(gender_model, "teacher_model_failure")) {
        failure_row(gender_model)
      } else {
        tidy_mixed(gender_model, outcome, "experience_gender_sensitivity")
      }
      row$GenderStratum <- gender
      gender_rows[[length(gender_rows) + 1]] <- row
    }
  }
  write_csv_utf8(
    bind_rows_fill(simple_rows),
    file.path(outdir, "08c_experience_simple_effects_Holm.csv")
  )
  write_csv_utf8(
    bind_rows_fill(boot_rows),
    file.path(outdir, "08d_experience_cluster_bootstrap_5000.csv")
  )
  write_csv_utf8(
    bind_rows_fill(failure_rows),
    file.path(outdir, "08e_experience_bootstrap_failures.csv")
  )
  write_csv_utf8(
    bind_rows_fill(gender_rows),
    file.path(outdir, "08f_experience_gender_sensitivity.csv")
  )
  all_diagnostics[[length(all_diagnostics) + 1]] <-
    experience$diagnostics
}

if (("H" %in% triggers || "I" %in% triggers) &&
    nzchar(aoi_path) && file.exists(aoi_path)) {
  aoi <- utils::read.csv(aoi_path, check.names = FALSE)
  aoi <- aoi[aoi$IncludedPrimary %in% TRUE, , drop = FALSE]
  for (column in c(
    "Participant", "WWR", "Complexity", "ExperienceGroup", "Gender",
    "OrderGroup", "AOICategory"
  )) {
    if (column %in% names(aoi)) aoi[[column]] <- factor(aoi[[column]])
  }
  if ("H" %in% triggers) {
    equipment <- aoi[
      aoi$Complexity == "C1" & aoi$AOICategory == "Equipment",
      , drop = FALSE
    ]
    equipment_rows <- list()
    for (outcome in intersect(
      c("Visited", "TFD", "TFD_ms", "AttentionShare"), names(equipment)
    )) {
      if (is.logical(equipment[[outcome]])) {
        equipment[[outcome]] <- as.integer(equipment[[outcome]])
      }
      fit <- fit_aoi_outcome(equipment, outcome, "equipment_C1")
      equipment_rows[[length(equipment_rows) + 1]] <- fit$fixed
      all_diagnostics[[length(all_diagnostics) + 1]] <- fit$diagnostic
    }
    write_csv_utf8(
      bind_rows_fill(equipment_rows),
      file.path(outdir, "09_equipment_C1_models.csv")
    )
  }
  if ("I" %in% triggers) {
    secondary <- intersect(
      c("Visited", "TFD", "TFD_ms", "TTFF", "TTFF_ms", "FC", "RFF", "MPD"),
      names(aoi)
    )
    secondary_rows <- list()
    for (aoi_name in c("Table", "Window")) {
      subset <- aoi[aoi$AOICategory == aoi_name, , drop = FALSE]
      for (outcome in secondary) {
        if (is.logical(subset[[outcome]])) {
          subset[[outcome]] <- as.integer(subset[[outcome]])
        }
        fit <- fit_aoi_outcome(
          subset, outcome, paste0("secondary_eye_", aoi_name)
        )
        row <- fit$fixed
        row$AOI <- aoi_name
        secondary_rows[[length(secondary_rows) + 1]] <- row
        all_diagnostics[[length(all_diagnostics) + 1]] <- fit$diagnostic
      }
    }
    write_csv_utf8(
      bind_rows_fill(secondary_rows),
      file.path(outdir, "10_secondary_eye_metrics.csv")
    )
  }
}

if ("J" %in% triggers && "S3" %in% names(input)) {
  s3_rows <- list()
  for (eye in c(
    "TableShare", "WindowShare", "RawCompetition", "AdjustedCompetition"
  )) {
    formula <- stats::as.formula(paste0(
      "S3 ~ ", eye,
      " + WWR * Complexity + ExperienceGroup + Gender + Block + ",
      "PositionWithinBlockCentered + OrderGroup + (1|Participant)"
    ))
    model <- fit_lmer_or_record(formula, input, "S3", paste0("S3_eye_", eye))
    row <- if (inherits(model, "teacher_model_failure")) {
      failure_row(model)
    } else {
      tidy_mixed(model, "S3", paste0("S3_eye_", eye))
    }
    row$EyePredictor <- eye
    s3_rows[[length(s3_rows) + 1]] <- row
  }
  write_csv_utf8(
    bind_rows_fill(s3_rows),
    file.path(outdir, "15_questionnaire_S3_eye_mixed_models.csv")
  )
}

write_csv_utf8(
  bind_rows_fill(all_diagnostics),
  file.path(outdir, "17_stage3_model_diagnostics.csv")
)
write_session_info(outdir)
