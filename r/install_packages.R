# Installs the R packages StatExplore needs into the project-local library (r/library).
# Usage (from the project root):  Rscript r/install_packages.R [phase]
args <- commandArgs(trailingOnly = TRUE)
phase <- if (length(args)) as.integer(args[[1]]) else 5L

packages <- list(
  `1` = c("jsonlite", "psych", "ggplot2"),
  `2` = c("car", "afex", "emmeans", "effectsize", "ppcor", "lmtest", "detectseparation"),
  `3` = character(),
  `4` = character(),
  `5` = c("lme4", "lmerTest", "nlme", "mmrm", "performance")
  # Optional, GitHub-only: remotes::install_github("dustinfife/flexplot", lib = "r/library")
)
wanted <- unique(unlist(packages[as.character(seq_len(phase))]))

script_dir <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
lib <- Sys.getenv("STATEXPLORE_R_LIB", file.path(script_dir, "library"))
dir.create(lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(lib, .libPaths()))

missing <- wanted[!vapply(wanted, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) {
  message("Installing: ", paste(missing, collapse = ", "), " into ", lib)
  install.packages(missing, lib = lib, repos = "https://cloud.r-project.org", type = "binary")
}
for (p in wanted) message(sprintf("%-10s %s", p, as.character(utils::packageVersion(p, lib.loc = .libPaths()))))
