# Code release validation

## Neutral names and prepared-input boundary: 2026-10-25

The package uses `reference_ais` for higher-frequency validation inputs, `vessel_uid` for the stable vessel key and `registry_imo` for the roster IMO identifier. The extraction entry is `pipeline/gfw_release/10_extract_reference_ais_observed.py`; the corresponding CLI options use `--reference-ais-root` and `--reference-ais-shards`. Users prepare and validate their own input adaptation. No column-mapping module or input-source adapter was added.

- `python src/check_code.py --tests`: **178 passed, 2 deselected, 45 dependency deprecation warnings**. All **313 source-file hashes, 33 stage symlinks and 289 Python sources** pass. The number of tests is unchanged.
- After applying the documented name substitutions to the pre-edit working-tree baseline, all **106 changed Python modules** have identical abstract syntax trees and all **5 changed configuration/provenance JSON files** compare equal. `source_manifest.json` is audited separately for its authorized path, hash and adaptation metadata changes. Numerical constants, filtering, joins and scientific calculations are unchanged; renamed fields require prepared inputs using the new schema.
- Every original `source_sha256` and research commit remains unchanged. Release hashes were updated for **128 manifest entries**, including aliases pointing to changed files. The single string-valued `adaptation` was converted to an `adaptations` list. Renamed historical source paths are explicitly marked as public aliases.
- The formal cleaner reads `p1_full/vessel_specs_by_mmsi_2025.csv`, SHA-256 `c5f4725a22d84778695cbbf5b9efbd9ff55ba9058b392c1b4ee75cf9c8466a2a`. A read-only server check of all **150,968 rows / 73 columns**, yielding **49,002 normalized vessel identities**, gives exactly equal normalized particulars and enriched type/speed outputs for the original research definitions and the previous release. The neutral-name release gives exactly the same results after audit-only column renaming in memory. Service speeds cover 40,534 identities; the unchanged type-default fallback covers 8,468. No source file was rewritten and no full AIS-cleaning run was performed. This establishes interface equivalence, not independent vessel-source verification.

### Historical exact-input gates

The two deselected tests and their required execution, validation and weather-configuration hashes remain unchanged. The bundled weather configuration SHA-256 is `fb6d2846a44034d3161928b851e66f68478e6ea8c491f50ca1b288ba3fd07532`, both before and after this naming update. The historical authorization requires `dece34f00777c390df2389c1b6601d86b8910f96ebedb0a21a8a825307a98302`. This pre-existing mismatch is not fixed by adding the absent annual receipts. Exact historical replay requires the matching original configuration and inputs; the gates have not been relaxed or rebound.

The frozen-weather asset manifest changes from `3a3c38fee94090c31fdb0152042d68ef94b853408e22ff42f861164f76345a5b` to `6cc5f8abfc5fed775c97a29f3525fcbe0e40e191f77923f4039b871d4fafd478` because its purpose text uses the neutral reference name. All individual asset hashes, sizes and metadata are unchanged. The two excluded tests do not bind the asset-manifest file hash; `verify_weather_assets()` continues to check the individual assets.

Existing frozen-roster and receipt hashes are also preserved. Inputs whose field names change have different bytes and require their own validated snapshot; these checks do not establish byte-for-byte replay with renamed inputs. No complete scientific rebuild, label replacement, dataset regeneration or publication occurred.

The following sections retain checks of earlier interface versions; the current naming update is documented above.

## Prepared-input schema and registry cleanup

The shared schema omits an unused observation-year provider lookup. The public input registry omits four unused acquisition-cache and pre-ingest aliases; its 41 retained mappings are unchanged. Original source hashes and the two packaging adaptations are recorded in `source_manifest.json`.

- `python src/check_code.py --tests`: **178 passed, 2 deselected**, with the same 45 dependency deprecation warnings.
- All **313 source hashes, 33 stage symlinks and 289 Python sources** pass the integrity and syntax checks.
- Apart from its module documentation and the removed unused lookup, the schema is syntax-tree identical. Its Arrow fields and types are identical, and an actual synthetic GFW shard written with the original and revised schemas is byte-for-byte identical, retaining `gfw_presence_v4.0` as its source label.
- The dataset CSVs, scientific parameters and retained source labels were not changed. No full scientific rebuild was run.

## Prepared-input release

The public controller now starts from prepared observation shards. Provider acquisition/recovery modules and the raw-response fleet-extraction entry are excluded. Data selection, cleaning, voyage construction, exposure calculation, cross-year observation and CHN aggregation remain in the analysis package.

The fresh public repository starts with this snapshot. Earlier validation records below describe predecessor packages; their Git history is not included in this repository.

- `python src/check_code.py --tests`: **178 passed, 2 deselected** in the same local Python 3.12.15 environment documented below. The suite emitted 45 dependency deprecation warnings. The two existing deselections still require external annual execution receipts.
- **313 source entries, 33 stage symlinks and 289 Python sources** pass the integrity and syntax checks. The deduplication function is syntax-tree identical to its former fleet-preparation implementation. All 33 unchanged functions/classes in the affected visit, harmonization, mask and deduplication-test modules match the preceding release; the documented entry-point and local-file changes are excluded from that comparison.
- Fourteen additional tests cover valid local inputs without acquisition receipts, missing/altered shards, incorrect row counts or month, invalid UTC timestamps, duplicate vessel-hours, wrong shard assignment, identity/roster changes, invalid coordinates, inconsistent totals, and a missing land-mask file. Invalid prepared buffers stop before a run directory is created. Valid-input and land-mask checks run with network connection attempts rejected.
- Original vessel-roster hashes, weather-input gates and January-baseline population checks remain in place. The new boundary verifies the supplied observations directly instead of invoking download or raw-response recovery modules.
- All **12 selected public CLI entries** load independently with `--help`, including the new `--buffer` and `--input-root` interfaces. Edited documentation links resolve, excluded acquisition modules are absent, and the stage execution/probe receipt function is syntax-tree identical to the preceding release.

The dataset CSVs and scientific output files were not changed or regenerated. Exporter byte parity below was established for the preceding code update; its exporter and aggregation kernels are unchanged in this update. No full scientific rebuild, provider request, data publication or input-source substitution was performed.

## Previous cross-year update: 2026-10-24 (`14ece94`)

Checked in the independent release checkout on `main`, importing packaged files rather than the research workspace. The update adds the adopted cross-year chain, its reader dependencies, CHN regrouping and the current field guide.

- `python src/check_code.py --tests`: **164 passed, 2 deselected**. The unchanged 2026-10-18 package was first reproduced with **143 passed, 2 deselected** in the same environment. New coverage includes January/February timestamps, the exclusive March cutoff, exclusion of 2026 departures, unchanged-voyage support, weather-calendar extension, acquisition closure, CHN vessel deduplication, separate city codes, and rejection of unmerged aggregates by the public wrapper.
- Integrity checks pass for **321 source entries, 34 relative symlinks and 296 Python sources**. Non-narrative syntax-tree comparison passes for all 25 newly copied Python modules, excluding only the documented disabled account-specific historical diagnostic planner. Scientific conditions, numerical constants and computation remain unchanged.
- All **11 formal follow-up/CHN entry points** load independently with `--help`, including fleet preparation and visit reconstruction; their transitive reader imports are included.
- In a temporary standalone checkout on the data server, the public exporter read the frozen E009 full-precision country and city aggregates. Its CSVs match the accepted files **byte for byte**: 122,770 rows/41 columns (25,661,461 bytes) and 579,836 rows/43 columns (130,995,161 bytes). No trajectory, weather or aggregation rebuild was run; frozen inputs and outputs were read only. Temporary server artifacts were removed.

| Frozen CSV | SHA-256 reproduced by the public exporter |
|---|---|
| `01_release_country_pair_month.csv` | `faef7495c96e423e77663fd34b6f9aacaca89e7347cd64a3e3e2ebeeb7c4fe22` |
| `02_release_city_pair_month.csv` | `2bfa6ef0c525304fc7a709e9334bf44cb9f94a368c7ab87a50baef9782a6f801` |

Local test environment: Python 3.12.15, NumPy 2.5.4, pandas 2.3.3, SciPy 1.18.1, PyArrow 26.0.0, xarray 2026.9.0, Shapely 2.2.0, NetworkX 3.7, PyYAML 6.0.3, scikit-learn 1.9.1 and pytest 9.1.1. Numerical-library threads were one. The suite emitted 46 dependency deprecation warnings; there were no test failures. Server CSV parity used the existing scientific environment.

No provider requests, complete scientific run or data publication occurred. These checks establish package integrity, selected synthetic behaviors and exact CSV export parity; they do not replace source-data validation or add independent validation of the new observation period. Source-bound historical receipts still require matching source bytes, as explained in [REPRODUCTION.md](REPRODUCTION.md).

## Prepared-particulars interface check: 2026-10-25

The cleaning interface now accepts the documented normalized columns for general type, detailed type and service speed. Three raw-response field aliases were removed in that check. The subsequent neutral-name update below also renames the internal identity fields. The original source hash and the adapted release hash are both recorded in `source_manifest.json`.

- `python src/check_code.py --tests`: **178 passed, 2 deselected, 45 dependency deprecation warnings**; 313 source-file hashes, 33 stage symlinks and 289 Python sources pass integrity and syntax checks.
- Every function in `cleaning_common.py` has an unchanged syntax tree. That earlier check loaded `p2/extended_sample_mmsi_2025.csv` (36,000 rows; SHA-256 `32526296cde85a63489de621f29880a48b78686391f0b3eef44f3e64c8d9c078`) and produced exactly equal normalized particulars and enriched type/speed outputs. It did not check the formal cleaning input, `p1_full/vessel_specs_by_mmsi_2025.csv`. That input is checked below.
- Input documentation now states its schemas directly. These packaging edits do not replace the vessel labels or their sources and do not regenerate the scientific outputs.

## Historical check: 2026-10-18

The following records the preceding package before the cross-year update.

## Completed checks

- `python src/check_code.py --tests`: **143 passed, 2 deselected**. The suite covers GFW cleaning and segmentation rules, voyage guards, wave bounds, cyclone and ERA5 matching, duration weights, track reconstruction, weighted aggregation, monthly regrouping, public export and the ten-minute review gates.
- Nine release-wrapper checks exercise country and city CSV/Parquet input, the 41/43-field schema, retention of single-voyage groups, the unclipped matched wave mean, missing wave values, removal of private and sampling fields, rejection of invalid groups/city codes/ranges, and preservation of existing files.
- 282 copied-file SHA-256 values and 17 relative stage-entry symlinks match the source manifest. All 258 Python files parse. Adapted files are recorded individually in the manifest; the adaptations remove conversational fields and research-server paths and leave scientific settings unchanged. No research-server path, server user name or other project name remains in the checkout.
- The full historical test tree (`pytest src/reference tests`) gives the same 9 failures and 84 errors before and after these adaptations; each needs an external receipt, data asset (for example the ocean mask) or optional package (`searoute`). The added review-gate module accounts for the 10 additional passing tests.
- The current checkout contains no final dataset CSVs, raw vessel/weather files, aggregate figure datasets or generated figures. The method's fixed anchorage-correction CSV remains under configuration. The repository history starts from this code-only state and contains no earlier dataset or figure artifacts.
- The standalone exporter `--help` works without loading the scientific dependency stack. Selected text files are checked for embedded credential/private-key patterns; none were found.

Two tests in `pipeline/gfw_release/tests/test_step5_authorization.py` require the original annual `execution.json`, validation receipt and original weather configuration: `test_user_authorization_binds_the_actual_gfw_baseline` and `test_changed_weather_configuration_is_rejected`. The selected runner deselects those two tests explicitly. Their assertions remain intact; the remaining scope-check tests run normally. The configuration hash mismatch described in the current check above predates the naming update, so supplying the missing receipts alone is not sufficient to run these two historical tests against the packaged configuration.

## Environment used

Python 3.10.19, NumPy 2.2.6, pandas 2.3.3, SciPy 1.15.3, PyArrow 22.0.0, xarray 2025.6.1, Shapely 2.1.2, NetworkX 3.4.2, PyYAML 6.0.3, scikit-learn 1.7.2 and pytest 8.4.2. Numerical-library threads were set to one. The selected checks do not require the netCDF4 backend. The earlier observed server environment remains recorded separately in `research_environment.json`.

No annual reconstruction, full monthly regeneration, data deposit or provider download was performed. These checks establish source-package integrity and the selected synthetic-data behaviors. Full processing still requires the external inputs, manifests and stage validation described in [REPRODUCTION.md](REPRODUCTION.md).
