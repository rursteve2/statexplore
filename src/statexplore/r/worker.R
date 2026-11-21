# StatExplore persistent R worker.
#
# Protocol: one JSON request per line on stdin; one response line on stdout,
# prefixed with "@@SX@@ ". Anything else written to stdout/stderr (package
# start-up messages, console printing) is treated by Python as log output.
#
# Request:  {"id": str, "cmd": "ping" | "versions" | "run" | "shutdown", "args": {...}}
#   run:    args = {"script": str, "packages": [str]}
#           The script is evaluated exactly as given in the persistent session
#           environment `.sx_env`. It must assign a list to `sx_result`.
# Response: {"id", "ok", "result", "error", "warnings", "elapsed"}

local({
  lib <- Sys.getenv("STATEXPLORE_R_LIB", "")
  if (nzchar(lib) && dir.exists(lib)) .libPaths(c(lib, .libPaths()))
})

if (!requireNamespace("jsonlite", quietly = TRUE)) {
  cat("@@SX@@ {\"id\":\"startup\",\"ok\":false,\"error\":\"R package 'jsonlite' is not installed.\"}\n")
  quit(save = "no", status = 3)
}

.sx_env <- new.env(parent = globalenv())

.sx_send <- function(x) {
  cat("@@SX@@ ", jsonlite::toJSON(x, auto_unbox = TRUE, null = "null", na = "null",
                                   digits = NA, force = TRUE), "\n", sep = "")
  flush(stdout())
}

.sx_versions <- function(pkgs) {
  pkgs <- unique(c("jsonlite", unlist(pkgs)))
  vers <- lapply(pkgs, function(p) {
    v <- tryCatch(as.character(utils::packageVersion(p)), error = function(e) NA_character_)
    v
  })
  names(vers) <- pkgs
  list(r_version = R.version.string, packages = vers, platform = R.version$platform)
}

.sx_run <- function(script, packages) {
  warns <- character()
  if (exists("sx_result", envir = .sx_env, inherits = FALSE)) rm("sx_result", envir = .sx_env)
  withCallingHandlers(
    eval(parse(text = script, keep.source = FALSE), envir = .sx_env),
    warning = function(w) {
      warns <<- c(warns, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )
  if (!exists("sx_result", envir = .sx_env, inherits = FALSE))
    stop("Script did not assign `sx_result`.")
  res <- get("sx_result", envir = .sx_env)
  list(result = res, warnings = warns, versions = .sx_versions(packages))
}

.sx_handle <- function(line) {
  req <- tryCatch(jsonlite::fromJSON(line, simplifyVector = FALSE), error = function(e) NULL)
  if (is.null(req) || is.null(req$id)) {
    .sx_send(list(id = "invalid", ok = FALSE, error = "Malformed request"))
    return(TRUE)
  }
  t0 <- proc.time()[["elapsed"]]
  out <- tryCatch({
    switch(req$cmd,
      ping = list(result = "pong", warnings = character()),
      versions = list(result = .sx_versions(req$args$packages), warnings = character()),
      run = .sx_run(req$args$script, req$args$packages),
      shutdown = list(result = "bye", warnings = character()),
      stop(sprintf("Unknown command: %s", req$cmd))
    )
  }, error = function(e) structure(list(message = conditionMessage(e)), class = "sx_error"))
  elapsed <- proc.time()[["elapsed"]] - t0
  if (inherits(out, "sx_error")) {
    .sx_send(list(id = req$id, ok = FALSE, error = out$message, elapsed = elapsed))
  } else {
    .sx_send(list(id = req$id, ok = TRUE, result = out$result, warnings = as.list(out$warnings),
                  versions = out$versions, elapsed = elapsed))
  }
  !identical(req$cmd, "shutdown")
}

.sx_main <- function() {
  con <- file("stdin", open = "r", encoding = "UTF-8")
  on.exit(close(con))
  .sx_send(list(id = "ready", ok = TRUE, result = .sx_versions(list())))
  repeat {
    line <- readLines(con, n = 1L, warn = FALSE)
    if (length(line) == 0L) break           # stdin closed: parent exited
    if (!nzchar(trimws(line))) next
    if (!.sx_handle(line)) break
  }
}

.sx_main()
