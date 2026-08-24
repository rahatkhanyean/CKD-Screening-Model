# Reproduction pipeline.
#
#   make reproduce   validate hashes -> analyses -> tests -> figures/tables
#                    -> manuscript -> verify manuscript matches results
#
# Individual targets are listed by `make help`. Long stages are separated so
# a reader can reproduce the cheap parts without the ~2 h of cross-validation.

PY ?= .venv/Scripts/python.exe
PDFLATEX ?= pdflatex

.DEFAULT_GOAL := help
.PHONY: help hashes prepare analyse cheap expensive tests tables figures \
        manuscript supplement verify reproduce clean syncfigs

help:
	@echo "Targets:"
	@echo "  hashes      verify input data checksums"
	@echo "  prepare     stages 0-2: provenance gate, cleaning, EDA"
	@echo "  cheap       all analyses that reuse committed CV predictions"
	@echo "  expensive   re-run nested cross-validation (~2 h)"
	@echo "  tests       full pytest suite"
	@echo "  tables      regenerate LaTeX table fragments"
	@echo "  syncfigs    copy analysis figures into paper/figures/"
	@echo "  manuscript  build paper/ieee_paper.pdf"
	@echo "  supplement  build paper/supplement.pdf"
	@echo "  verify      check manuscript numbers against generated results"
	@echo "  reproduce   hashes -> prepare -> cheap -> tables -> tests ->"
	@echo "              manuscript -> supplement -> verify"

hashes:
	@$(PY) -c "import hashlib,sys; \
	expected={'data/raw/ckd-dataset-v2.csv':'f24075f420b0f271bfddf3844a40061dbe9e2cb1c2336daa64463122344ea84a', \
	'data/external/uci2015/raw/data.csv':'0eeea8d17f5ad8792d854999d6ddf1602ec4e116616f6ccdfdaef0ba8109e694'}; \
	bad=[p for p,h in expected.items() if hashlib.sha256(open(p,'rb').read()).hexdigest()!=h]; \
	sys.exit('CHECKSUM MISMATCH: '+', '.join(bad)) if bad else print('input hashes OK')"

prepare: hashes
	$(PY) scripts/00_provenance.py
	$(PY) scripts/01_prepare_data.py
	$(PY) scripts/02_eda.py

# Analyses that reuse the committed cross-validation predictions.
cheap: prepare
	$(PY) scripts/04_evaluate.py
	$(PY) scripts/07_literature.py
	$(PY) scripts/09_optimism.py
	$(PY) scripts/11_continuous_recovery.py
	$(PY) scripts/13_tripod_checklist.py
	$(PY) scripts/17_provenance_figure.py
	$(PY) scripts/18_mimic_etl.py
	$(PY) scripts/20_benchmark_diagnostics.py
	$(PY) scripts/21_provenance_sensitivity.py
	$(PY) scripts/22_build_feature_registry.py
	$(PY) scripts/25_bootstrap_convergence.py

# Stages that refit models. Hours, not minutes.
expensive:
	$(PY) scripts/03_nested_cv.py
	$(PY) scripts/05_importance_stability.py
	$(PY) scripts/10_ebm_shapes.py
	$(PY) scripts/12_binned_vs_continuous.py
	$(PY) scripts/14_encoding_robustness.py
	$(PY) scripts/15_procedure_bootstrap.py --resamples 200
	$(PY) scripts/19_external_validation.py
	$(PY) scripts/23_casemix_reanalysis.py

tests:
	$(PY) -m pytest tests/ -q

tables:
	$(PY) scripts/24_make_latex_tables.py

figures:
	@echo "figures are produced by their owning analysis stages (see cheap/expensive)"

# The build reads paper/figures/; the analysis stages write reports/figures/.
# Without this the manuscript can be built from a stale copy of a figure.
syncfigs:
	$(PY) -c "import reproduce, sys; sys.exit(0 if reproduce.sync_figures() else 1)"

manuscript: tables syncfigs
	cd paper && $(PDFLATEX) -interaction=nonstopmode -halt-on-error ieee_paper.tex >/dev/null
	cd paper && $(PDFLATEX) -interaction=nonstopmode -halt-on-error ieee_paper.tex >/dev/null
	@echo "built paper/ieee_paper.pdf"

supplement: tables syncfigs
	cd paper && $(PDFLATEX) -interaction=nonstopmode -halt-on-error supplement.tex >/dev/null
	cd paper && $(PDFLATEX) -interaction=nonstopmode -halt-on-error supplement.tex >/dev/null
	@echo "built paper/supplement.pdf"

verify:
	$(PY) -m pytest tests/test_manuscript_consistency.py -q

reproduce: hashes prepare cheap tables syncfigs tests manuscript supplement verify
	@echo ""
	@echo "Reproduction complete. Manuscript numbers verified against"
	@echo "generated results. Note: 'reproduce' reuses the committed"
	@echo "cross-validation predictions; run 'make expensive' first to"
	@echo "regenerate them from scratch."

clean:
	rm -f paper/*.aux paper/*.log paper/*.out paper/*.toc
