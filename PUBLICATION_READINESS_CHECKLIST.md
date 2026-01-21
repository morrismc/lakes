# Publication Readiness Checklist

**Project:** Lake Density Analysis
**Target:** High-impact journal (Nature Geoscience, Science, PNAS)
**Status:** Pre-submission preparation

---

## Scientific Rigor

### Data Quality
- [x] Lake data validated (NHD quality checks)
- [x] Glacial boundaries from peer-reviewed sources
- [x] Environmental rasters standardized to common CRS
- [x] Area calculations validated (<1% error)
- [ ] Contact USGS for NHD mapping metadata by region
- [ ] Document any data exclusions with justification

### Statistical Methods
- [x] Multiple testing correction implemented (FDR)
- [x] Spatial autocorrelation tested (Moran's I)
- [x] Model comparison performed (exponential vs. alternatives)
- [x] Detection bias modeled and corrected
- [x] Sensitivity analysis comprehensive (thresholds, grid sizes)
- [ ] Homoscedasticity tests (Levene's test)
- [ ] Normality tests (Shapiro-Wilk, Q-Q plots)
- [ ] Power analysis for hypothesis tests
- [ ] Posterior predictive checks for Bayesian models

### Assumptions
- [x] All assumptions documented (SCIENTIFIC_ASSUMPTIONS.md)
- [x] Validation status specified for each assumption
- [x] Violated assumptions addressed with mitigations
- [ ] Update assumption status after running all validation tests
- [ ] Create supplementary table summarizing assumptions

---

## Code Quality

### Documentation
- [x] All modules have docstrings
- [x] Key functions have examples
- [x] Scientific rationale documented
- [x] API organized (high-level vs. low-level functions)
- [ ] Add missing docstrings to helper functions
- [ ] Standardize docstring format (NumPy style)
- [ ] Create Sphinx-compatible documentation

### Testing
- [x] Manual validation of key results
- [ ] Unit tests for critical functions
- [ ] Integration tests for pipelines
- [ ] Regression tests (reproducibility)
- [ ] Set up continuous integration (GitHub Actions)

### Error Handling
- [x] Statistical functions have proper error handling
- [ ] Fix bare `except:` clauses (6 instances identified)
- [ ] Add input validation to all public functions
- [ ] Informative error messages
- [ ] Logging instead of print statements

### Code Organization
- [x] Modules clearly separated by functionality
- [x] No circular dependencies
- [x] Configuration centralized
- [ ] Refactor 200+ line functions into smaller pieces
- [ ] Extract duplicate code into utilities
- [ ] Add type hints (Python 3.10+)

---

## Figures and Tables

### Main Text Figures
- [ ] Figure 1: Study design and map (300 DPI, colorblind-safe)
- [ ] Figure 2: Half-life analysis (4-panel)
- [ ] Figure 3: Multivariate analysis (4-panel)
- [ ] Figure 4: Detection bias and validation (4-panel)
- [ ] All figures: Check font sizes (8-12pt), legend clarity, axis labels

### Supplementary Figures
- [ ] S1: Power law analysis
- [ ] S2: Spatial patterns
- [ ] S3: Size-stratified half-life
- [ ] S4: Bayesian diagnostics
- [ ] S5: Spatial autocorrelation
- [ ] S6: Grid size sensitivity
- [ ] All figures: Consistent style, high resolution

### Tables
- [ ] Table 1: Summary statistics by glacial stage
- [ ] Table 2: Bayesian model results
- [ ] Table 3: Multivariate regression results
- [ ] Table 4: Model comparison (AIC, BIC)
- [ ] Table S1: Complete regression outputs
- [ ] Table S2: Sensitivity analysis summary
- [ ] All tables: Clear captions, units specified

---

## Manuscript Sections

### Title and Abstract
- [ ] Draft 3-5 title options
- [ ] Abstract <150 words (if Nature/Science) or <250 (if PNAS)
- [ ] Abstract includes: context, question, approach, result, implication
- [ ] No references in abstract

### Introduction
- [ ] Hook: Why lake distribution matters
- [ ] Context: Davis's hypothesis, previous work
- [ ] Gap: Lack of quantitative continental-scale test
- [ ] Question: Does lake density decay exponentially with age?
- [ ] Approach: Bayesian + multivariate framework
- [ ] Length: 3-4 paragraphs for Nature/Science, 5-6 for PNAS

### Methods
- [ ] Study region and data sources
- [ ] Glacial chronosequence analysis (Bayesian model)
- [ ] Multivariate statistical analysis (variance partitioning)
- [ ] Detection bias and sensitivity analysis
- [ ] Statistical inference (multiple testing, spatial autocorrelation)
- [ ] All methods justified with citations
- [ ] Reproducibility: Code and data availability statement
- [ ] Length: ~2000-3000 words

### Results
- [ ] Result 1: Exponential decay with t½ ~660 ka
- [ ] Result 2: Variance partitioning (topography vs. glaciation)
- [ ] Result 3: Detection bias impacts magnitude but not pattern
- [ ] Result 4: Results robust to model and parameter choices
- [ ] Each result tied to specific figure panel
- [ ] Statistics reported: effect sizes, CIs, p-values (corrected)
- [ ] Length: ~1500-2000 words

### Discussion
- [ ] Integration: Temporal + spatial perspectives converge
- [ ] Mechanism: Glaciation → topography → lakes → decay
- [ ] Comparison: Previous estimates (qualitative → quantitative)
- [ ] Implications: Limnology, geomorphology, paleoecology
- [ ] Limitations: Detection bias, limited stages, CONUS-specific
- [ ] Future directions: European/Canadian replication, point processes
- [ ] Length: ~1500-2000 words

### Supplementary Information
- [ ] Extended methods
- [ ] Supplementary figures
- [ ] Supplementary tables
- [ ] Supplementary discussion (if needed)
- [ ] Code and data availability

---

## Data and Code Sharing

### Data Archive (Zenodo or similar)
- [ ] Raw NHD data (or link to source)
- [ ] Glacial boundary shapefiles
- [ ] Environmental rasters
- [ ] Derived datasets (gridded densities, classifications)
- [ ] README with data dictionary
- [ ] License (CC-BY-4.0 recommended)

### Code Repository (GitHub)
- [ ] Clean repository (remove diagnostic scripts)
- [ ] README with installation instructions
- [ ] requirements.txt or environment.yml
- [ ] Example notebooks demonstrating key analyses
- [ ] License (MIT or GPL)
- [ ] Citation information (CITATION.cff)
- [ ] Release version (tag with DOI)

---

## Reviewer Anticipation

### Methodological Concerns Addressed
- [x] Detection bias modeled and corrected
- [x] Spatial autocorrelation tested
- [x] Multiple testing correction applied
- [x] Model validation (exponential vs. alternatives)
- [x] Sensitivity analysis comprehensive
- [ ] Prepare responses to anticipated questions (see CO_AUTHOR_GUIDE.md)

### Alternative Explanations Considered
- [x] Climate vs. glaciation disentangled (multivariate analysis)
- [x] Topography as mediator (variance partitioning)
- [ ] Post-glacial lake formation discussed (beaver ponds, etc.)
- [ ] Temporal climate changes acknowledged

### Limitations Acknowledged
- [x] Detection bias creates magnitude uncertainty
- [x] Limited temporal replication (3 stages + NADI-1)
- [x] Age-climate confound acknowledged
- [x] CONUS-specific, generalizability unknown
- [ ] Clearly state in Discussion section

---

## Submission Preparation

### Journal Selection
- [ ] Identify 3-5 target journals in priority order
- [ ] Review journal scope and recent lake/geomorphology papers
- [ ] Check word limits, figure limits, formatting requirements
- [ ] Identify potential editors and reviewers

### Pre-Submission Inquiry (Optional)
- [ ] Draft 1-paragraph summary of findings
- [ ] Send to editor of top-choice journal
- [ ] Gauge interest before full submission

### Cover Letter
- [ ] Introduce research question and significance
- [ ] Highlight novelty (quantitative test of Davis's hypothesis)
- [ ] Emphasize rigor (Bayesian, multivariate, robustness checks)
- [ ] Suggest editors (if applicable)
- [ ] Suggest reviewers (and exclusions, if conflicts)

### Author Contributions
- [ ] CRediT taxonomy statement
- [ ] All authors approve final manuscript
- [ ] Authorship order agreed upon
- [ ] Corresponding author designated

### Competing Interests
- [ ] Declare any financial or non-financial competing interests
- [ ] (Typically "none" for basic research)

### Ethics and Permissions
- [ ] Data use permissions (NHD, glacial boundaries)
- [ ] No human subjects (N/A)
- [ ] No endangered species (N/A)

---

## Post-Submission

### Revision Plan
- [ ] Prepare for major revisions (typical for high-impact journals)
- [ ] Assign tasks for common revision requests
- [ ] Timeline for rapid turnaround (2-4 weeks)

### Response to Reviewers
- [ ] Point-by-point response document
- [ ] Mark changes in manuscript
- [ ] Additional analyses if requested
- [ ] Polite and professional tone

---

## Timeline

**Week 1-2:**
- Complete validation tests
- Generate final figures
- Update SCIENTIFIC_ASSUMPTIONS.md

**Week 3-4:**
- Write Methods section
- Write Results section
- Create supplementary materials

**Week 5-6:**
- Write Introduction and Discussion
- Internal review by co-authors
- Revise based on feedback

**Week 7-8:**
- Final polishing
- Format for target journal
- Submit!

**Target Submission Date:** [TO BE DETERMINED]

---

## Key Contacts

**Lead Author:** [Name, Email]
**Co-Authors:** [List]
**Statistical Consultant:** [If applicable]
**Data Sources:**
- USGS NHD: [Link]
- NADI-1: Dalton et al. (2020)

---

## Sign-Off

**Lead Author:** ☐ Ready for submission
**Co-Author 1:** ☐ Approved
**Co-Author 2:** ☐ Approved
**Co-Author N:** ☐ Approved

---

**Last Updated:** 2026-01-21
**Status:** In preparation
**Estimated Completion:** [Date]
