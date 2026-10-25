# Code and data publication

GitHub hosts code, configuration, tests and documentation. The current source includes the cross-year follow-up through February 2026 and the CHN country regrouping. The MIT licence applies to this code.

The public entry point starts from prepared local inputs. Provider queries, downloads and raw-response recovery are outside this release; [INPUTS.md](INPUTS.md) describes the analysis inputs. This repository begins with a code-only snapshot of that workflow. Source provenance and packaging adaptations are recorded in [source_manifest.json](source_manifest.json); the manuscript describes the inputs used to generate the study's results.

The dataset deposit is a separate Zenodo draft, with reserved DOI `10.5281/zenodo.23453528`. The DOI becomes active after publication. This code update does not publish the dataset. Its archive version and data licence remain to be finalized in the deposit metadata; the code's MIT licence does not set the dataset's terms.

The upload candidate comprises two monthly aggregate CSVs (122,770 country records with 41 fields; 579,836 coded city records with 43 fields), a README, a field guide and a checksum manifest. Both tables cover the same 1,136,916 voyages departing in UTC 2025, using observation through February 2026.

Raw AIS/provider responses, vessel identities, voyage-level data and city-name mappings are excluded from the public dataset. They are also absent from this code repository. Its history began with a code-only snapshot; no earlier commercial-AIS table or generated figure was committed.
