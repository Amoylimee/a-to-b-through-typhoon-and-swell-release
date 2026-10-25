# Maritime cyclone and wave exposure — GFW 2025 code

Scientific code accompanying **Global tropical cyclone and wave exposure across maritime origin–destination pairs in 2025**. The workflow uses Global Fishing Watch (GFW) vessel positions to construct voyages, reconstruct tracks, match tropical cyclones and waves, quantify missing-position exposure ranges, and aggregate results by departure month.

The public workflow starts from **prepared local inputs**: vessel positions, vessel particulars, anchorages and environmental data. Provider queries, downloads, account setup and raw-response recovery are outside this repository. [Input specifications](docs/INPUTS.md) describe the files needed by the analysis.

This GitHub repository contains code, configuration, tests and documentation. **Final datasets will be archived separately in Zenodo.** The reserved dataset DOI is `10.5281/zenodo.23453528`; it becomes active after publication. Dataset files, aggregate plotting data and generated figures are not included in this checkout.

## Start here

Use Python 3.10 or later. Source integrity checks use the standard library:

```bash
python3 src/check_code.py
```

For the selected scientific and export tests, create the environment and run:

```bash
conda env create -f environment.yml
conda activate shipping-exposure-release
python src/check_code.py --tests
```

The tests use synthetic inputs and do not download vessel or weather data. See [validation results](docs/VALIDATION.md) for checks completed on this release and the two annual-input tests that require external run records.

## Current workflow and output schema

The current code includes the adopted cross-year observation update: UTC 2025 departures with arrival and destination-visit closure observed before 1 March 2026. The resulting sample contains 1,136,916 voyages from 34,190 vessels, with 122,770 country records and 579,836 coded city records. The finite observation window does not guarantee completion of every long voyage.

- Each voyage contributes entirely to its UTC departure month, including time in a subsequent month.
- Durations and exposure quantities are summed before calculating means and shares. Groups with one voyage are retained.
- The country table has 41 fields; the coded city table has 43. City-name lookup tables and vessel identities are excluded from the public export.
- HKG, TWN and MAC map to CHN. Country measures are reaggregated from voyage statistics, with distinct-vessel counts; cities retain separate codes.
- `mean_swh_matched_m` is the primary wave mean and remains unchanged when it lies outside the missing-position range. `mean_swh_constrained_m` is a secondary comparison field.
- Data-availability measures and missing-position ranges are retained. Sampling-interval fields are excluded from the current public schema.

[Reproduction instructions](docs/REPRODUCTION.md) map the code to each method, including the cross-year entry point that accepts a prepared `--buffer`. The [GFW field guide](docs/GFW_FIELD_GUIDE.md) describes the output fields; it does not imply that the tables are stored here.

Given an external **full-precision monthly aggregate**, export the current public columns with:

```bash
python src/build_public_table.py --level country \
  --input /path/to/country_full_precision.parquet \
  --output /path/to/new-country.csv
```

Use `--level city` for an aggregate with fixed `origin_city_code` and `dest_city_code` columns. CSV input is also supported. Inputs must already contain the CHN regrouping; the exporter rejects unmerged HKG/TWN/MAC records. It validates the aggregate, applies publication rounding and preserves missing values. It requires a new output path. It does not reconstruct voyages or reaggregate previously rounded tables.

## Provenance and access

The cross-year and CHN update is sourced from research commit `ec9159837fc036c4baec7892c73851b895416895`, with accepted implementation point `14cab7e5407f0ec9384da947699210185ce5ddaf`. Unchanged kernels retain their earlier source provenance. [source_manifest.json](docs/source_manifest.json) records each source path, commit, hash and packaging adaptation. The public controller validates prepared inputs and starts with visit reconstruction; scientific methods and settings in the downstream stages are preserved. Relative stage symlinks and historical monthly aliases share one implementation. See [reproduction instructions](docs/REPRODUCTION.md) for external inputs, source-hash gates and packaging changes.

[Publication status](docs/PUBLICATION.md) tracks the separate code and data deposits. Source code retains the existing [MIT licence](LICENSE); citation metadata are in [CITATION.cff](CITATION.cff).
