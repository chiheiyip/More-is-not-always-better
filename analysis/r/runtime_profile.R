# Registered locale controls factor ordering and numerical summation order.
configure_locked_runtime <- function(lock_path, profile) {
  if (!requireNamespace("jsonlite", quietly = TRUE)) stop("Locked jsonlite unavailable")
  profiles <- jsonlite::fromJSON(lock_path, simplifyVector = FALSE)$profiles
  if (profile == "auto") {
    matches <- names(profiles)[vapply(profiles, function(p) identical(p$R, as.character(getRversion())), logical(1))]
    if (length(matches) != 1L) stop("Unregistered R version")
    profile <- matches[1]
  }
  settings <- profiles[[profile]]
  if (is.null(settings)) stop("Unregistered R runtime profile")
  if (!identical(settings$R, as.character(getRversion())) || !identical(settings$platform, R.version$platform))
    stop("Locked R version/platform mismatch")
  for (p in names(settings$packages)) {
    actual <- tryCatch(as.character(packageVersion(p)), error=function(e) "not-installed")
    if (!identical(actual, settings$packages[[p]])) stop("Locked R package mismatch: ", p)
  }
  for (category in names(settings$locale)) {
    actual <- Sys.setlocale(category, settings$locale[[category]])
    if (!identical(actual, settings$locale[[category]]))
      stop("Locked R locale unavailable: ", category, ": ", settings$locale[[category]])
  }
  Sys.setenv(TZ = settings$timezone)
  invisible(list(locale = setNames(lapply(names(settings$locale), Sys.getlocale), names(settings$locale)),
                 timezone = Sys.getenv("TZ")))
}
