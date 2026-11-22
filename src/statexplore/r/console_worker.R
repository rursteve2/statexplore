# StatExplore R Console worker: runs the USER's own R code in a process separate from the
# analysis workers. Same JSON-lines protocol as worker.R (responses prefixed "@@SX@@ ").
#
# Commands:
#   load      {script}          run StatExplore's data-load script privately; expose a copy as `my_data`
#   run_user  {code, plot_dir, width, height}
#                               parse and evaluate the code expression by expression in the user
#                               environment; capture printed output, messages, warnings, errors and plots
#   objects / packages          workspace inventory
#   clear                       remove the user's objects and restore `my_data`
#   export    {csv, meta}       write `my_data` (CSV + column types) for an explicit import
#   shutdown
#
# This is NOT a sandbox: user code runs with the user's permissions.

local({
  core <- Sys.getenv("STATEXPLORE_R_LIB", ""); user <- Sys.getenv("STATEXPLORE_USER_LIB", "")
  libs <- c(if (nzchar(core) && dir.exists(core)) core, if (nzchar(user) && dir.exists(user)) user)
  # core library first: user-installed packages can never mask StatExplore's own dependencies
  .libPaths(c(libs, .libPaths()))
})
if (!requireNamespace("jsonlite", quietly = TRUE)) {
  cat("@@SX@@ {\"id\":\"startup\",\"ok\":false,\"error\":\"R package 'jsonlite' is not installed.\"}\n")
  quit(save = "no", status = 3)
}
options(repos = NULL)  # never contact CRAN implicitly

.sxc_env <- new.env(parent = globalenv())       # the user's workspace
.sxc_original <- NULL                           # pristine copy of my_data

.sxc_send <- function(x) {
  cat("@@SX@@ ", jsonlite::toJSON(x, auto_unbox = TRUE, null = "null", na = "null", digits = I(15), force = TRUE),
      "\n", sep = "")
  flush(stdout())
}

.sxc_load <- function(script) {
  e <- new.env(parent = globalenv())
  eval(parse(text = script, keep.source = FALSE), envir = e)
  d <- e$dat
  d$sx_split <- NULL
  names(d) <- unname(e$sx_names[names(d)])
  .sxc_original <<- d
  assign("my_data", d, envir = .sxc_env)
  list(rows = nrow(d), cols = ncol(d))
}

.sxc_run <- function(code, plot_dir, width, height) {
  entries <- list()
  add <- function(type, text) if (length(text) && any(nzchar(text)))
    entries[[length(entries) + 1]] <<- list(type = type, text = paste(text, collapse = "\n"))
  exprs <- tryCatch(parse(text = code, keep.source = TRUE), error = function(e) e)
  if (inherits(exprs, "error")) {
    add("error", paste("Syntax error:", conditionMessage(exprs)))
    return(list(entries = entries, plots = I(character()), error = TRUE))
  }
  dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)
  before <- dev.list()
  grDevices::png(file.path(plot_dir, "sxplot%03d.png"), width = width, height = height, units = "in", res = 110)
  dev_id <- dev.cur()
  failed <- FALSE
  srcs <- attr(exprs, "srcref")
  for (i in seq_along(exprs)) {
    add("input", if (!is.null(srcs)) paste(as.character(srcs[[i]]), collapse = "\n") else deparse(exprs[[i]]))
    res <- tryCatch(withCallingHandlers({
        out <- utils::capture.output(v <- withVisible(eval(exprs[[i]], envir = .sxc_env)), type = "output")
        add("output", out)
        if (v$visible) add("output", utils::capture.output(print(v$value), type = "output"))
        NULL
      },
      message = function(m) { add("message", sub("\n$", "", conditionMessage(m))); invokeRestart("muffleMessage") },
      warning = function(w) { add("warning", paste("Warning:", conditionMessage(w))); invokeRestart("muffleWarning") }),
      error = function(e) e)
    if (inherits(res, "error")) {
      call <- conditionCall(res)
      if (is.call(call) && identical(call[[1]], quote(eval))) call <- NULL   # top level: not the console's own eval()
      add("error", paste0("Error", if (!is.null(call)) paste0(" in ", paste(deparse(call), collapse = " ")), ": ",
                          conditionMessage(res)))
      failed <- TRUE
      break
    }
  }
  for (d in setdiff(dev.list(), before)) try(dev.off(d), silent = TRUE)
  plots <- list.files(plot_dir, pattern = "^sxplot\\d+\\.png$", full.names = TRUE)
  plots <- plots[file.info(plots)$size > 0]
  list(entries = entries, plots = I(plots), error = failed)   # I(): one plot must still be a JSON array
}

.sxc_objects <- function() {
  nm <- ls(.sxc_env)
  if (!length(nm)) return(list())
  lapply(nm, function(n) { o <- get(n, envir = .sxc_env)
    list(name = n, class = paste(class(o), collapse = "/"),
         size = if (!is.null(dim(o))) paste(dim(o), collapse = " x ") else as.character(length(o))) })
}

.sxc_packages <- function() {
  attached <- sub("^package:", "", grep("^package:", search(), value = TRUE))
  loaded <- loadedNamespaces()
  lapply(sort(union(attached, loaded)), function(p) list(package = p, attached = p %in% attached,
    version = tryCatch(as.character(utils::packageVersion(p)), error = function(e) NA_character_),
    library = tryCatch(dirname(find.package(p)), error = function(e) NA_character_)))
}

.sxc_export <- function(csv, meta) {
  if (!exists("my_data", envir = .sxc_env)) stop("There is no `my_data` object in the workspace.")
  d <- get("my_data", envir = .sxc_env)
  if (!is.data.frame(d)) stop("`my_data` is not a data frame.")
  info <- lapply(names(d), function(n) { x <- d[[n]]
    list(name = n, class = class(x)[1], levels = if (is.factor(x)) levels(x) else NULL,
         ordered = is.ordered(x)) })
  out <- d
  for (n in names(out)) {
    x <- out[[n]]
    if (is.factor(x)) out[[n]] <- as.character(x)
    else if (inherits(x, "Date") || inherits(x, "POSIXt")) out[[n]] <- format(x)
    else if (is.logical(x)) out[[n]] <- as.integer(x)
  }
  utils::write.csv(out, csv, row.names = FALSE, na = "", fileEncoding = "UTF-8")
  writeLines(jsonlite::toJSON(list(rows = nrow(d), columns = info), auto_unbox = TRUE, null = "null"), meta,
             useBytes = TRUE)
  list(rows = nrow(d), cols = ncol(d))
}

.sxc_handle <- function(line) {
  req <- tryCatch(jsonlite::fromJSON(line, simplifyVector = FALSE), error = function(e) NULL)
  if (is.null(req) || is.null(req$id)) { .sxc_send(list(id = "invalid", ok = FALSE, error = "Malformed request")); return(TRUE) }
  t0 <- proc.time()[["elapsed"]]
  a <- req$args
  out <- tryCatch(switch(req$cmd,
      ping = "pong",
      load = .sxc_load(a$script),
      run_user = .sxc_run(a$code, a$plot_dir, a$width %||% 7, a$height %||% 5),
      objects = .sxc_objects(),
      packages = .sxc_packages(),
      clear = { rm(list = ls(.sxc_env, all.names = TRUE), envir = .sxc_env)
                if (!is.null(.sxc_original)) assign("my_data", .sxc_original, envir = .sxc_env); "cleared" },
      export = .sxc_export(a$csv, a$meta),
      shutdown = "bye",
      stop(sprintf("Unknown command: %s", req$cmd))),
    error = function(e) structure(list(message = conditionMessage(e)), class = "sx_error"))
  el <- proc.time()[["elapsed"]] - t0
  if (inherits(out, "sx_error")) .sxc_send(list(id = req$id, ok = FALSE, error = out$message, elapsed = el))
  else .sxc_send(list(id = req$id, ok = TRUE, result = out, warnings = list(), elapsed = el))
  !identical(req$cmd, "shutdown")
}

local({
  con <- file("stdin", open = "r", encoding = "UTF-8")
  on.exit(close(con))
  .sxc_send(list(id = "ready", ok = TRUE, result = list(r_version = R.version.string, libs = .libPaths())))
  repeat {
    line <- readLines(con, n = 1L, warn = FALSE)
    if (length(line) == 0L) break
    if (!nzchar(trimws(line))) next
    if (!.sxc_handle(line)) break
  }
})
