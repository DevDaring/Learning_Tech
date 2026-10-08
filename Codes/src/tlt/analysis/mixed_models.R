# Mixed models for Section 6.6, called by stage 9 (analysis) with:
#   Rscript mixed_models.R <input.csv> <output_dir> <block>
# Input columns: item_id, model, condition, PQI, and one column per rubric dimension
# (ordinal labels coded 0 = best category ... k = worst; categorical dimensions arrive
# as 0/1 indicators named <dim>__bad).
#
#   LMM:   PQI ~ condition * model + (1 | item_id) + (1 | item_id:model)          lme4::lmer
#   CLMM:  dim ~ condition * model + (1 | item_id) + (1 | item_id:model)          ordinal::clmm
#   GLMM:  dim__bad ~ condition * model + (1 | item_id) + (1 | item_id:model)     lme4::glmer
# Omnibus tests of the condition main effect and the condition x model interaction are
# likelihood-ratio tests against the nested model (ML fits). Coefficients are written in
# full. Coding: treatment coding for condition (reference EN in Block A, C0 in Block B) and
# sum-to-zero coding for producer, so each condition coefficient is the effect averaged
# over producers. CLMM sign: outcomes are coded 0 = best category, so a POSITIVE CLMM
# condition coefficient means WORSE ratings than the reference (opposite to PQI).

suppressPackageStartupMessages({
  library(lme4)
  library(ordinal)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
inp <- args[1]; outdir <- args[2]; block <- args[3]
d <- read.csv(inp, stringsAsFactors = FALSE)
ref <- if (block == "A") "EN" else "C0"
lv <- unique(d$condition)
d$condition <- factor(d$condition, levels = c(ref, setdiff(sort(lv), ref)))
d$model <- factor(d$model)
contrasts(d$model) <- contr.sum(nlevels(d$model))
d$item_id <- factor(d$item_id)
d$item_model <- interaction(d$item_id, d$model, drop = TRUE)

coef_rows <- list()
test_rows <- list()

# Likelihood-ratio test computed from the two log-likelihoods, so it works identically for
# lmer (ML fits), glmer and clmm (anova() cannot compare clmm objects inside a function).
lrt <- function(full, reduced, name, outcome, family) {
  if (is.null(full) || is.null(reduced)) {
    return(data.frame(block = block, outcome = outcome, family = family, test = name,
                      chisq = NA_real_, df = NA_real_, p_value = NA_real_, note = "fit failed"))
  }
  l1 <- logLik(full); l0 <- logLik(reduced)
  df <- attr(l1, "df") - attr(l0, "df")
  chisq <- as.numeric(2 * (l1 - l0))
  data.frame(block = block, outcome = outcome, family = family, test = name, chisq = chisq,
             df = df, p_value = pchisq(max(chisq, 0), df, lower.tail = FALSE), note = "")
}

add_coefs <- function(fit, outcome, family) {
  s <- summary(fit)$coefficients
  df <- data.frame(block = block, outcome = outcome, family = family, term = rownames(s),
                   estimate = s[, 1], se = s[, 2], row.names = NULL)
  df$z_or_t <- s[, 3]
  df$p_value <- if (ncol(s) >= 4) s[, 4] else 2 * pnorm(-abs(df$z_or_t))
  df
}

# ---- LMM on PQI
dp <- d[!is.na(d$PQI), ]
f_full <- PQI ~ condition * model + (1 | item_id) + (1 | item_model)
fit <- lmer(f_full, data = dp, REML = TRUE)
coef_rows[[length(coef_rows) + 1]] <- add_coefs(fit, "PQI", "lmm")
ml_full <- lmer(f_full, data = dp, REML = FALSE)
ml_noint <- lmer(PQI ~ condition + model + (1 | item_id) + (1 | item_model), data = dp, REML = FALSE)
ml_nocond <- lmer(PQI ~ model + (1 | item_id) + (1 | item_model), data = dp, REML = FALSE)
test_rows[[length(test_rows) + 1]] <- lrt(ml_full, ml_noint, "condition:model", "PQI", "lmm")
test_rows[[length(test_rows) + 1]] <- lrt(ml_noint, ml_nocond, "condition (main)", "PQI", "lmm")
vc <- as.data.frame(VarCorr(fit))
write.csv(vc, file.path(outdir, paste0("lmm_variance_block", block, ".csv")), row.names = FALSE)

# ---- CLMM per ordinal dimension and GLMM per categorical indicator
# Every fit (full, no interaction, no condition) is an independent task, run in parallel with
# forked workers; the models, formulas and data are exactly those of a sequential run.
ncores <- as.integer(Sys.getenv("TLT_R_CORES", "6"))
ord_dims <- grep("^ord__", names(d), value = TRUE)
bad_dims <- grep("__bad$", names(d), value = TRUE)
rhs <- c(full = "condition * model", noint = "condition + model", nocond = "model")
tasks <- list()
for (col in c(ord_dims, bad_dims)) for (k in names(rhs)) tasks[[length(tasks) + 1]] <- list(col = col, k = k)
fit_task <- function(t) {
  col <- t$col
  dd <- d[!is.na(d[[col]]), ]
  if (length(unique(dd[[col]])) < 2) return(NULL)
  re <- "+ (1 | item_id) + (1 | item_model)"
  tryCatch({
    if (grepl("^ord__", col)) {
      dd$y <- factor(dd[[col]], ordered = TRUE)
      clmm(as.formula(paste("y ~", rhs[[t$k]], re)), data = dd, nAGQ = 1)
    } else {
      ctl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
      glmer(as.formula(paste(col, "~", rhs[[t$k]], re)), data = dd, family = binomial, control = ctl)
    }
  }, error = function(e) { message(col, " ", t$k, ": ", conditionMessage(e)); NULL })
}
fits <- parallel::mclapply(tasks, fit_task, mc.cores = ncores, mc.preschedule = FALSE)
get_fit <- function(col, k) {
  for (i in seq_along(tasks)) if (tasks[[i]]$col == col && tasks[[i]]$k == k) {
    f <- fits[[i]]
    return(if (inherits(f, "try-error")) NULL else f)
  }
  NULL
}
for (col in c(ord_dims, bad_dims)) {
  fam <- if (grepl("^ord__", col)) "clmm" else "glmm"
  name <- if (fam == "clmm") sub("^ord__", "", col) else col
  full <- get_fit(col, "full")
  if (is.null(full)) next
  coef_rows[[length(coef_rows) + 1]] <- add_coefs(full, name, fam)
  red <- get_fit(col, "noint"); red2 <- get_fit(col, "nocond")
  test_rows[[length(test_rows) + 1]] <- lrt(full, red, "condition:model", name, fam)
  test_rows[[length(test_rows) + 1]] <- lrt(red, red2, "condition (main)", name, fam)
}

write.csv(do.call(rbind, coef_rows), file.path(outdir, paste0("mixed_coefficients_block", block, ".csv")), row.names = FALSE)
tr <- do.call(rbind, test_rows)
if (!is.null(tr)) write.csv(tr, file.path(outdir, paste0("mixed_tests_block", block, ".csv")), row.names = FALSE)
cat("done\n")
