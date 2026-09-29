# ==========================================================================
#  Computer Vision project — build orchestration
#  Deliverables: proposal doc, final report, presentation, experiments.
# ==========================================================================

LATEXMK      := latexmk -xelatex -interaction=nonstopmode -halt-on-error
PROPOSAL_DIR := docs/proposal
REPORT_DIR   := docs/report
SLIDES_DIR   := slides

.PHONY: all docs proposal report slides experiments clean clean-tex help

all: docs            ## Build every document (default)

docs: proposal report slides   ## Build proposal + report + slides

## --- Documents -----------------------------------------------------------
proposal:            ## Compile the project proposal -> docs/proposal/proposal.pdf
	cd $(PROPOSAL_DIR) && $(LATEXMK) proposal.tex

report:              ## Compile the final report (if present)
	@if [ -f $(REPORT_DIR)/report.tex ]; then \
		cd $(REPORT_DIR) && $(LATEXMK) report.tex; \
	else echo "[skip] $(REPORT_DIR)/report.tex not present yet"; fi

slides:              ## Compile the presentation (Beamer, if present)
	@if [ -f $(SLIDES_DIR)/slides.tex ]; then \
		cd $(SLIDES_DIR) && $(LATEXMK) slides.tex; \
	else echo "[skip] $(SLIDES_DIR)/slides.tex not present yet"; fi

## --- Experiments ---------------------------------------------------------
experiments:         ## Run the full experiment pipeline (see experiments/README.md)
	@if [ -f experiments/run_all.py ]; then \
		uv run python experiments/run_all.py; \
	else echo "[skip] experiments/run_all.py not present yet"; fi

## --- Housekeeping --------------------------------------------------------
clean-tex:           ## Remove LaTeX build artifacts, keep PDFs
	-cd $(PROPOSAL_DIR) && latexmk -c 2>/dev/null || true
	-cd $(REPORT_DIR)   && latexmk -c 2>/dev/null || true
	-cd $(SLIDES_DIR)   && latexmk -c 2>/dev/null || true

clean: clean-tex     ## Remove all build artifacts including PDFs
	-cd $(PROPOSAL_DIR) && latexmk -C 2>/dev/null || true
	-cd $(REPORT_DIR)   && latexmk -C 2>/dev/null || true
	-cd $(SLIDES_DIR)   && latexmk -C 2>/dev/null || true

help:                ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
