# AI-enabled Detection of Exoplanets from Noisy TESS Light Curves

## Methodology
We developed a hybrid physics-informed and AI-driven pipeline for detecting and classifying transit-like dips in noisy TESS light curves. The pipeline ingests SAP/PDCSAP light curves, applies TESS quality masking, selects the safest flux source, normalizes the flux, performs conservative detrending, and records quality-control metrics including cadence, time baseline, gaps, robust noise, CROWDSAP, FLFRCSAP, and centroid availability. Synthetic fallback is disabled for real data so failed downloads cannot produce artificial successes.

Periodic dips are detected using a BLS/TLS-style search. For every detected candidate, the pipeline estimates period, epoch, duration, depth, SNR, number of observed transits, and periodogram strength. Detection is separated from classification: a periodic dip can be a planet, eclipsing binary, blend, starspot-like variability, instrumental systematic, or uncertain signal.

Candidate parameters are refined with a local transit-window fit and event-by-event depth estimates. The vetting module extracts physically meaningful features: odd/even depth mismatch, secondary-eclipse significance at phase 0.5, centroid-shift significance, crowding/dilution risk, V-shape score, harmonic risk, red-noise proxy, and data-quality score. These features are passed to both a transparent rule-based classifier and an optional supervised AI classifier trained on curated labeled candidate catalogs.

## Classification, significance, and uncertainty
The classifier outputs probabilities for planetary transit candidate, eclipsing binary, blend/contaminated signal, stellar variability, instrumental/systematic, no significant signal, and uncertain transit-like signal. Physical guardrails prevent the AI model from overcalling planet candidates when strong secondary eclipses, odd/even mismatch, centroid motion, or poor data quality are present.

Signal significance is reported through local transit SNR, periodogram strength, and effective SNR after red-noise inflation. Parameter uncertainties combine local photometric scatter, residual bootstrap depth uncertainty, event-to-event depth scatter, ephemeris-grid curvature, and multi-detrender stability. Final confidence is a weighted triage confidence using detection confidence, parameter confidence, and classification confidence. It is not claimed to be a formal astronomical validation probability.

## Validation and final outputs
The validation framework supports curated labeled datasets and synthetic injection-recovery experiments. Detection is evaluated with precision, recall, specificity, and F1. Classification is evaluated with accuracy, balanced accuracy, macro F1, weighted F1, and confusion matrices. Parameter recovery is evaluated through period, duration, and depth errors against injected or curated ground truth. Confidence calibration is checked with reliability bins comparing reported confidence with empirical correctness.

The final sector-scale system writes per-target cache files, resume-safe batch outputs, failure logs, target summaries, raw candidate catalogs, harmonized final candidate catalogs, class-distribution plots, confidence plots, period-depth-priority plots, and a top-candidate review table. The final catalog includes period, duration, depth, SNR, uncertainty estimates, predicted class, confidence level, false-positive risk indicators, and recommended follow-up action.

## Assumptions and limitations
We assume periodic dips are approximately stable over the observed baseline and that detrending windows are longer than the transit duration. Transit depth can be biased by dilution in crowded apertures; therefore CROWDSAP is treated as a risk feature rather than a hard rejection. Planet radius is only meaningful when reliable stellar radius is available. Crowded-field and high-priority candidates should receive follow-up checks using target pixel files, Gaia nearby-source information, and difference imaging before being treated as validated planets.
