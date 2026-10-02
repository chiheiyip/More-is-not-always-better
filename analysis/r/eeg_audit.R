# Standalone validation suite. No promotion or access to historical outputs.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4) stop("usage: eeg_audit.R input.csv outdir outcomes.csv design.json")
file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
source(file.path(dirname(normalizePath(sub("^--file=", "", file_arg[[1]]))), "common.R"))
assert_packages()
if (!requireNamespace("jsonlite", quietly = TRUE)) stop("jsonlite is required")
# Mark UTF-8 strings without transcoding to the Windows process locale.
# fileEncoding would truncate Chinese identifiers under a legacy C locale.
input <- withCallingHandlers(
  read.csv(args[[1]], check.names = FALSE, na.strings = c("", "NA"), encoding = "UTF-8"),
  warning = function(w) stop("Input decoding failed: ", conditionMessage(w))
)
names(input)[[1]] <- sub("^\ufeff", "", names(input)[[1]])
outdir <- args[[2]]
spec <- read.csv(args[[3]], stringsAsFactors = FALSE)
design <- jsonlite::fromJSON(args[[4]])
set.seed(design$seed)
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
for (field in names(design$factor_levels)) {
  values <- as.character(input[[field]])
  levels <- design$factor_levels[[field]]
  if (any(!is.na(values) & !values %in% levels)) stop(paste("Unknown factor level:", field))
  input[[field]] <- factor(values, levels = levels)
  contrasts(input[[field]]) <- contr.treatment(levels, base = 1)
}
input$Participant <- factor(input$Participant)
base <- "WWR * Complexity + WWR * ExperienceGroup + Complexity * ExperienceGroup + Gender"
rhs <- list(
  Model0 = paste(base, "+ (1|Participant)"),
  Model1 = paste(base, "+ Block + PositionWithinBlockCentered + OrderGroup + (1|Participant)"),
  PreviousScene = paste(base, "+ Block + PositionWithinBlockCentered + OrderGroup + PreviousWWR + PreviousComplexity + (1|Participant)"),
  Block1 = paste(base, "+ PositionWithinBlockCentered + OrderGroup + (1|Participant)")
)
coef_rows <- list(); diagnostics <- list(); sample_rows <- list()
for (i in seq_len(nrow(spec))) {
  outcome <- spec$outcome[[i]]
  suites <- if (spec$core[[i]]) c("Model1", "Model0", "PreviousScene", "Block1") else "Model1"
  for (suite in suites) {
    formula <- as.formula(paste(outcome, "~", rhs[[suite]]))
    data <- input
    if (suite == "PreviousScene") data <- data[data$PositionWithinBlock > 1, ]
    if (suite == "Block1") data <- data[data$Block == 1, ]
    data <- data[complete.cases(data[, all.vars(formula), drop = FALSE]), , drop = FALSE]
    if (suite == "Model1" && nrow(data) != nrow(input)) {
      stop("R Model1 dropped rows from the validated complete sample; check decoding and design")
    }
    if (nrow(data)) sample_rows[[length(sample_rows)+1]] <- data.frame(
      outcome=outcome, model=suite, Participant=data$Participant,
      GlobalTrialOrder=data$GlobalTrialOrder
    )
    warnings <- character()
    model <- tryCatch(withCallingHandlers(
      lme4::lmer(formula, data=data, REML=FALSE),
      warning=function(w) { warnings <<- c(warnings, conditionMessage(w)); invokeRestart("muffleWarning") }
    ), error=function(e) e)
    row <- data.frame(outcome=outcome, scale=spec$scale[[i]], model=suite,
      formula=paste(deparse(formula), collapse=""), n_trials=nrow(data),
      n_participants=length(unique(data$Participant)), fit_status="fit_failed",
      converged=FALSE, singular=NA, rank_deficient=NA, cr2_status="not_run",
      optimizer_code=NA_character_, diagnostic=paste(warnings, collapse="; "))
    if (inherits(model, "error")) {
      row$diagnostic <- paste(row$diagnostic, conditionMessage(model), sep="; ")
    } else {
      opt <- model@optinfo
      row$fit_status <- "fit"
      row$optimizer_code <- paste(opt$conv$opt, collapse=",")
      messages <- opt$conv$lme4$messages
      row$converged <- all(opt$conv$opt == 0) && length(messages) == 0
      row$singular <- lme4::isSingular(model)
      row$rank_deficient <- length(attr(lme4::getME(model, "X"), "col.dropped")) > 0
      row$diagnostic <- paste(c(warnings, messages), collapse="; ")
      robust <- tryCatch({
        frame <- model.frame(model)
        V <- clubSandwich::vcovCR(model, cluster=frame$Participant, type="CR2")
        tests <- clubSandwich::coef_test(model, vcov=V, test="Satterthwaite")
        intervals <- clubSandwich::conf_int(model, vcov=V, test="Satterthwaite", level=0.95)
        data.frame(outcome=outcome, scale=spec$scale[[i]], model=suite,
          term=rownames(tests), estimate=tests$beta, std.error=tests$SE,
          df=tests$df_Satt, p.value=tests$p_Satt,
          CI_low=intervals$CI_L, CI_high=intervals$CI_U,
          n_trials=nobs(model), n_participants=length(unique(frame$Participant)))
      }, error=function(e) e)
      if (inherits(robust, "error")) {
        row$cr2_status <- "failed"
        row$diagnostic <- paste(row$diagnostic, conditionMessage(robust), sep="; ")
      } else {
        row$cr2_status <- "computed"
        coef_rows[[length(coef_rows)+1]] <- robust
      }
    }
    diagnostics[[length(diagnostics)+1]] <- row
  }
}
coefficient_table <- if (length(coef_rows)) bind_rows_fill(coef_rows) else data.frame(
  outcome=character(), scale=character(), model=character(), term=character(),
  estimate=numeric(), std.error=numeric(), df=numeric(), p.value=numeric(),
  CI_low=numeric(), CI_high=numeric(), n_trials=integer(), n_participants=integer()
)
sample_table <- if (length(sample_rows)) bind_rows_fill(sample_rows) else data.frame(
  outcome=character(), model=character(), Participant=character(), GlobalTrialOrder=integer()
)
write_csv_utf8(coefficient_table, file.path(outdir,"coefficients.csv"))
write_csv_utf8(bind_rows_fill(diagnostics), file.path(outdir,"diagnostics.csv"))
write_csv_utf8(sample_table, file.path(outdir,"model_samples.csv"))
write_session_info(outdir)
