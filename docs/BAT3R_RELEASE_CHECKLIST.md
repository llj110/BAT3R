# BAT3R release audit — 2026-10-12

## Addressed in this update

- BAT3R implementations and runnable entry points now have BAT3R filenames.
- `bat3r` is accepted by all four existing model update dispatch sites;
  `ttt3r` remains a legacy alias with its actual meaning documented.
- The GitHub version of `model.py` ended inside its executable example's
  unclosed constructor. The missing tail was recovered from the server copy.
- Documentation uses `llj110/BAT3R`, real command-line flags, and the upstream
  CUT3R checkpoint source. Placeholder paper/project links were removed.
- ImageIO, scikit-image, Open3D, and evo dependencies are explicitly listed.
- Checkpoint storage outside `src` no longer prevents finding repository code.
- Compatibility imports, CPU regression checks, and source-check CI are included.
- README license text now refers to the existing notices rather than claiming
  Apache-2.0, which is not the license text bundled in this repository.

## Still needed for a complete research release

1. **Checkpoint provenance.** Publish the exact checkpoint used for reported
   BAT3R results, with a checksum, origin, and loading configuration. The
   upstream CUT3R checkpoint is a starting point and does not by itself establish
   the provenance of any added BAT3R parameters.
2. **Reproduction package.** The server has separate rebuttal scripts, runtime
   patches, seeds, and saved results outside this repository. Review and package
   the exact configurations and numerical verification needed for paper tables.
   Historical reports are not a fresh rerun of this renamed release.
3. **Environment lock.** Record a tested dependency lock and CUDA/driver versions.
   The current requirements include unpinned upstream libraries.
4. **Additional public materials.** The formal title, nine-author list, NeurIPS
   record, OpenReview URL, BibTeX, and CITATION.cff are now included. A dedicated
   project website, confirmed arXiv identifier, and result gallery remain to be
   published or linked.
5. **Baseline provenance.** Clearly separate upstream CUT3R/TTT3R runs from the
   modified repository's legacy mode names and runtime patches.
6. **Dataset protocol.** Release sequence lists, sampling lengths, preprocessing
   settings, and aggregation conventions for each reported benchmark.

7. **Name disambiguation.** A separate ECCV 2026 work uses the acronym BAT3R
   for articulated-object reconstruction. Keep this project's full paper title
   visible and review search/discovery branding before further public releases.
   The other work is [Bootstrapping Articulated 3D Reconstruction from 2D Image Collections](https://github.com/jakubzadrozny/bat3r).

Existing third-party license notices have been preserved. This update does not
replace those notices with a new project-wide license.
