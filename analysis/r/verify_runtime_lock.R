args <- commandArgs(TRUE)
if (length(args) != 2L) stop("Expected lock file and profile")
if (!requireNamespace("jsonlite", quietly = TRUE)) stop("Locked jsonlite unavailable")
lock <- jsonlite::fromJSON(args[1], simplifyVector = FALSE)
profile <- args[2]
if (profile == "auto") {
  matches <- names(lock$profiles)[vapply(lock$profiles, function(p)
    identical(p$R, as.character(getRversion())), logical(1))]
  if (length(matches) != 1L) stop("Unregistered R version: ", getRversion())
  profile <- matches[1]
}
expected <- lock$profiles[[profile]]
if (is.null(expected)) stop("Unknown R lock profile: ", profile)
errors <- character()
if (!identical(expected$R, as.character(getRversion())))
  errors <- c(errors, paste("R expected", expected$R, "found", getRversion()))
if (!identical(expected$platform, R.version$platform))
  errors <- c(errors, paste("R platform expected", expected$platform, "found", R.version$platform))
actual <- setNames(lapply(names(expected$packages), function(p) {
  tryCatch(as.character(utils::packageVersion(p)), error = function(e) "not-installed")
}), names(expected$packages))
for (p in names(expected$packages)) if (!identical(expected$packages[[p]], actual[[p]]))
  errors <- c(errors, paste(p, "expected", expected$packages[[p]], "found", actual[[p]]))
if (length(errors)) stop(paste(errors, collapse = "; "))
cat("ANALYSIS_RUNTIME_LOCK=", jsonlite::toJSON(list(status = "passed", profile = profile,
    R = as.character(getRversion()), platform = R.version$platform,
    packages = actual, library_paths = .libPaths()), auto_unbox = TRUE), "\n", sep = "")
