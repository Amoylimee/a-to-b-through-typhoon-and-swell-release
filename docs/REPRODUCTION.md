# Scientific source and reproduction

## Code and external data

This is the GFW 2025 code repository, including cross-year follow-up through February 2026. The dataset has reserved Zenodo DOI `10.5281/zenodo.23453528`, which becomes active after publication. Raw inputs, per-voyage results and city-name mappings are excluded from that public deposit and from this repository. Tests here use synthetic data and do not request provider data.

Processing begins with local, prepared input files. Data acquisition and provider-specific response handling are external to this release. See [INPUTS.md](INPUTS.md) for schemas, preparation rules and the prepared observation-buffer interface.

Users implement and check their own input adaptation before this boundary. The public code consumes the documented schema and supplies the subsequent analysis stages. The higher-frequency validation interface uses `reference_ais`; roster identities use `vessel_uid` and `registry_imo`.

Packaged provenance registries use public path aliases for renamed reference files, including paths in `PROMOTION.json`. In `source_manifest.json`, `source_path_kind: "public_alias"` identifies the renamed source path; `source_sha256` and `research_commit` still identify the original research bytes, while `release_sha256` identifies the packaged file. The original path mapping is retained in the internal review record. Public aliases are not claims that the renamed path existed at the historical commit. See [VALIDATION.md](VALIDATION.md) for the limits of historical receipt replay.

The reference tree is `src/reference/02_coding/`. Keeping the `02_coding` level preserves relative imports and root calculations. `pipeline/gfw_release/` contains the established GFW methods; `pipeline/gfw_followup/` contains the cross-year stages, and `pipeline/gfw_monthly_release/` the shared monthly and CHN export code. Numbered stage entries are relative symlinks. Historical E008 monthly paths are compatibility symlinks to the shared implementation. Shared reconstruction, weather and E002 aggregation/bound kernels remain at their original relative paths.

## Stage map

Paths in this table are relative to `src/reference/02_coding/`.

| Method | Entry point or implementation |
|---|---|
| Harmonize prepared GFW presence | `pipeline/10_ingest_data/40_build_gfw_presence_shards/10_build_shards.py` |
| Clean GFW presence | `pipeline/20_clean_and_harmonize/30_clean_gfw_presence/10_clean_shards.py` |
| Port visits, city support and short-stop uncertainty | `pipeline/30_build_analysis_sample/60_build_gfw_voyages/10_identify_port_calls.py` |
| Construct and filter voyages | `pipeline/30_build_analysis_sample/60_build_gfw_voyages/20_build_voyages.py` |
| Reconstruct positions and match weather | `pipeline/40_construct_variables/70_construct_gfw_weather/10_build_tracks_and_weather.py` |
| Cyclone interpolation, ERA5 matching and duration weights | `pipeline/40_construct_variables/60_construct_voyage_weather_facts/weather_core/` |
| Exposure bounds and annual source assembly | `pipeline/50_run_main_analysis/20_build_gfw_release/10_build_release.py` |
| Position, weather, segmentation and interval validation | `pipeline/60_run_validation_and_robustness/30_validate_gfw_release/` |
| Annual intermediate documentation and summaries | `pipeline/80_package_final_outputs/30_package_gfw_release/` |
| Monthly regrouping and fixed city codes | `pipeline/gfw_monthly_release/monthly_release.py` and `10_build_monthly_release/10_build_tables.py` |
| Independent monthly counts and schema checks | `experiments/E008_gfw_monthly_public_release/20_validate_and_package/10_validate_and_document.py` |
| Remove sampling-interval fields | `experiments/E008_gfw_monthly_public_release/40_finalize_public_release/10_remove_sampling_fields.py` |
| Follow-up controller | `pipeline/00_environment_and_config/40_run_gfw_followup/10_run_followup.py` |
| Validate prepared follow-up observations | `pipeline/gfw_followup/prepared_inputs.py` |
| Cross-year visits, 2025 departures and frozen quality rules | `pipeline/30_build_analysis_sample/70_build_gfw_followup_voyages/` |
| Extend tracks and weather | `pipeline/40_construct_variables/80_construct_gfw_followup_weather/10_build_tracks_and_weather.py` |
| Follow-up bounds, monthly aggregation and CHN export | `pipeline/50_run_main_analysis/30_build_gfw_monthly_followup/10_build_bounds_and_release.py` |
| Censoring, monthly comparison and pending-visit audits | `pipeline/60_run_validation_and_robustness/40_audit_gfw_followup/` |
| Shared CHN regrouping/export functions | `pipeline/gfw_monthly_release/20_finalize_public_release/10_merge_china_regions.py` |

The current bounds stage calls the CHN exporter directly. HKG, TWN and MAC map to CHN before country aggregation; vessel counts are independently deduplicated. Physical cities remain distinct, existing public codes stay fixed, and new cities receive additional codes. The standalone CHN module's `main()` retains E008 input/count checks for historical replay; it is not the current follow-up driver. Output definitions are in [GFW_FIELD_GUIDE.md](GFW_FIELD_GUIDE.md).

## Cross-year observation

The current sample includes UTC 2025 departures with arrival and destination-visit closure observed before 1 March 2026: 1,136,916 voyages, 34,190 vessels and 150,305,435 voyage-hours, giving 122,770 country records and 579,836 coded city records. Follow-up observations retain their actual 2026 timestamps; 2026 departures are excluded. The finite window can still truncate long voyages.

The follow-up chain takes the prepared February observation buffer, frozen 2025 and January intermediate products, weather manifests and source partitions referenced by the track component bridge. The controller checks all 64 prepared shards against their hashes, counts, UTC month, unique vessel-hours and frozen vessel identities before creating a run. Weather and January-baseline gates remain unchanged. Each build stage still runs a probe before the complete stage. Its February boundary is explicit; individual stage defaults retain January settings for historical replay.

After providing those external inputs and matching manifests, invoke the controller from `src/reference/`:

```bash
python 02_coding/pipeline/00_environment_and_config/40_run_gfw_followup/10_run_followup.py \
  --buffer /path/to/prepared-february-buffer \
  --assets /path/to/frozen-february-weather-assets \
  --out 02_coding/.runs/new-february-rebuild --workers 16
```

Resolve historical input roots in the controller/stages and validate their bindings first. Prepared-data checks replace the download-receipt gate at this public entry point; downstream source, population, probe and weather gates remain intact. Unknown exposure stays unknown; no proportional adjustment fills missing voyages. Acquisition modules and the fleet-extraction command are excluded. The shared vessel-hour deduplication rule remains available in `prepared_inputs.py`, with its original regression tests. The land-mask utility requires its source file locally and does not download missing inputs.

## Monthly export

`src/build_public_table.py` exports an external full-precision monthly aggregate to the final 41-field country schema or 43-field coded city schema. It calls the existing E008 schema checks and CSV exporter, then the accepted sampling-field projection. It keeps single-voyage groups, the matched wave mean, missing values, flags and rounding conventions. It neither recalculates exposure nor clips the primary mean.

```bash
python src/build_public_table.py --help
python src/build_public_table.py --level country \
  --input /path/to/country_full_precision.parquet --output /path/to/new-country.csv
python src/build_public_table.py --level city \
  --input /path/to/city_full_precision.parquet --output /path/to/new-city.csv
```

Inputs may be Parquet or CSV and must contain the final fields at full calculation precision, with the CHN regrouping already applied. City inputs must use fixed public city codes. The wrapper rejects unmerged HKG/TWN/MAC records: relabelling or summing rounded rows cannot reproduce distinct-vessel counts and weighted exposure estimates. Additional columns are removed by the public allowlist. Sampling-interval fields need not be supplied. Output paths must be new, and an invalid aggregate fails before a public file is created. Export only after upstream input and scientific validation have passed.

The historical E008 full-build sequence first generated 50/52-field monthly tables, including sampling intervals, and then projected them to the current 41/43-field schema. Its original field-guide resource and bootstrap code remain for that sequence. Annual intermediate code similarly retains its historical five-voyage threshold. These settings describe intermediate products; the final monthly export retains all groups with at least one voyage and has no sampling-interval columns.

## Inputs for upstream reconstruction

Full processing requires prepared GFW presence and anchorage inputs, vessel particulars and type assignments, navigation/ocean references, ERA5 significant wave height and the recorded IBTrACS snapshot. Higher-frequency AIS is required for the external position/weather validation. None of those datasets is bundled. Input schemas and preparation rules are specified in [INPUTS.md](INPUTS.md). The small anchorage-correction CSV under `gfw_release/config/` is a fixed method configuration, not an exposure table or a public city-code lookup.

The annual drivers preserve snapshot hashes, expected counts, historical run names and acceptance gates. Input paths in the configurations are relative to the research root, `src/reference/`; run stage entries from that directory or point the paths to your inputs. Their default private run records and intermediate files are not included, so this checkout is not a one-command annual rebuild. Obtain the required inputs and frozen manifests, resolve each stage's paths, and check the supplied data against the recorded bindings before running. Existing scientific gates have not been bypassed. No annual run or new provider request was made during this code update.

The historical E008 monthly builder accepts `--source`, `--run`, `--workers`, `--probe` and `--resume`. `--source` is the validated annual run with its full-precision voyage statistics. Private outputs must stay under `src/reference/02_coding/.runs/`. A successful `<run>-probe` with matching input and code hashes is required before `<run>` can be built. This sequence replays the earlier sample; use the follow-up controller for the current product. Consult each entry's `--help`. Do not publish the generated `private/` directory or its city-name mapping.

## Environment and source integrity

`environment.yml` lists the scientific Python stack. `requirements.txt` provides a pip alternative for the self-contained tests and monthly exporter. Cleaning and port-call stages additionally use `sinbue` and `iogenius`; routing may use `searoute`, and retained input utilities may use `movingpandas` and `natsort`. The observed research environment in `research_environment.json` records provenance, not a fully tested lockfile for a fresh installation. [VALIDATION.md](VALIDATION.md) records the environment actually used for this update.

`source_manifest.json` records the source commit and SHA-256 for every copied reference file. Most files are byte-identical to their sources. For each adapted file the manifest records the original and packaged hashes and the change: nine JSON configurations omit conversational quotation fields; research-server paths are made relative to the research root, and other source repositories appear as `<upstream-*-repository>` placeholders; the GFW release driver starts child stages with the current Python interpreter; two E008 review-note sentences no longer attribute decisions to a conversation. Scientific settings and historical input hashes are preserved. The historical E006 step-5 weather authorization binds the original weather-configuration bytes, so that gate rejects the packaged configuration, as it would any configuration pointed to new input locations. Old run receipts still identify the original source bytes; a newly packaged hash must not be substituted into an old receipt.

```bash
python src/check_code.py
python src/check_code.py --tests
```

Source entries additionally record the cross-year integration commit and original/packaged hashes. Conversational receipt strings are neutralized. The prepared-input update removes provider acquisition/recovery modules, extracts the unchanged deduplication function, and changes the public controller to begin with validated local observation shards. The source manifest records these adaptations and excluded upstream files. Reproduction with relocated inputs or packaged source requires corresponding new validated manifests; never substitute packaged hashes into a historical receipt.

The first command checks copied-file hashes, relative stage symlinks and Python syntax. The second also runs the selected scientific, cross-year, CHN and CLI tests. Two exact-source annual integration tests are explicitly deselected because their frozen execution receipt is external; their assertions remain unchanged in the source tree. Running every historical test without the required upstream artifacts is not supported by the selected-test command.
