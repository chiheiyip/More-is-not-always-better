required_packages <- c(
  "lme4", "glmmTMB", "emmeans", "clubSandwich", "broom.mixed"
)

assert_packages <- function() {
  missing <- required_packages[
    !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
  ]
  if (length(missing) > 0) {
    stop(
      "Missing locked R packages: ", paste(missing, collapse = ", "),
      ". Run renv::restore() from analysis/r."
    )
  }
}

write_session_info <- function(outdir) {
  writeLines(
    capture.output(utils::sessionInfo()),
    file.path(outdir, "R_session_info.txt"),
    useBytes = TRUE
  )
}

write_csv_utf8 <- function(x, path) {
  utils::write.csv(x, path, row.names = FALSE, na = "", fileEncoding = "UTF-8")
}

bind_rows_fill <- function(items) {
  if (!length(items)) return(data.frame())
  columns <- unique(unlist(lapply(items, names), use.names = FALSE))
  normalized <- lapply(items, function(item) {
    missing <- setdiff(columns, names(item))
    for (column in missing) item[[column]] <- NA
    item[columns]
  })
  do.call(rbind, normalized)
}

tidy_mixed <- function(model, outcome, model_name) {
  result <- broom.mixed::tidy(model, effects = "fixed", conf.int = TRUE)
  result$outcome <- outcome
  result$model <- model_name
  result$status <- "fit"
  result$formula <- paste(deparse(stats::formula(model)), collapse = "")
  result
}

cr2_rows <- function(model, outcome, model_name, cluster) {
  model_frame <- stats::model.frame(model)
  if ("Participant" %in% names(model_frame)) {
    cluster <- model_frame$Participant
  }
  result <- clubSandwich::coef_test(
    model, vcov = "CR2", cluster = cluster, test = "Satterthwaite"
  )
  data.frame(
    outcome = outcome,
    model = model_name,
    term = rownames(result),
    estimate = result$beta,
    std.error = result$SE,
    df = result$df_Satt,
    p.value = result$p_Satt,
    stringsAsFactors = FALSE
  )
}

fit_lmer_or_record <- function(formula, data, outcome, model_name) {
  tryCatch(
    lme4::lmer(formula, data = data, REML = FALSE),
    error = function(e) structure(
      list(
        outcome = outcome, model = model_name,
        error = conditionMessage(e), formula = deparse(formula)
      ),
      class = "teacher_model_failure"
    )
  )
}

failure_row <- function(x) {
  data.frame(
    outcome = x$outcome, model = x$model, status = "fit_failed",
    diagnostic = x$error, formula = paste(x$formula, collapse = ""),
    stringsAsFactors = FALSE
  )
}

bh_within_family <- function(frame, family_columns) {
  if (nrow(frame) == 0 || !"p.value" %in% names(frame)) return(frame)
  key <- interaction(frame[family_columns], drop = TRUE, lex.order = TRUE)
  frame$p.value.BH <- ave(frame$p.value, key, FUN = function(x) p.adjust(x, "BH"))
  frame
}
