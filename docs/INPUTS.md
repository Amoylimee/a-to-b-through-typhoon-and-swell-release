# Prepared inputs

The public analysis starts from local files. Obtain and organize vessel positions, vessel particulars, anchorages, routing references and environmental inputs before running the stages. Provider API requests, credentials, download scheduling and raw-response recovery are outside this package.

Users prepare and validate their own source-to-schema conversion. The release starts after that conversion, using the fields, types and units below; it does not supply a configurable column mapper or a source-specific adapter.

Record each input's source, version, observation period, identifiers and checksum with the input snapshot. Exact replay uses the frozen study inputs and their recorded hashes; an alternative input snapshot is a new analysis and requires corresponding validation.

## Vessel positions

Prepared GFW presence has one retained position per vessel and UTC hour:

| Field | Meaning |
|---|---|
| `mmsi` | Canonical MMSI, positive integer |
| `hour_utc` | Timezone-aware UTC timestamp at the start of the represented hour |
| `lat`, `lon` | Latitude and longitude in decimal degrees |

Exact repeats at the same 0.01-degree cell retain one position. A vessel-hour with conflicting cells retains no position. The unchanged rule is available as `deduplicate()` in `src/reference/02_coding/pipeline/gfw_followup/prepared_inputs.py`; regression tests cover float32 cell precision and conflicting hours. Do not fill absent hours. Downstream harmonization places hourly observations at the hour midpoint and preserves unknown fields.

For the annual harmonizer, supply `fleet_presence_unique/2025-MM.parquet` for all twelve months and `fleet_presence_unique_summary.json`. The summary supplies `output_rows` and an `outputs` mapping from each filename to its `sha256`. Invoke `pipeline/10_ingest_data/40_build_gfw_presence_shards/10_build_shards.py` with `--input-root` and a new `--out` under `02_coding/.runs/`. The older `--acquisition-root` spelling remains an alias for this local-file argument.

## Vessel particulars and identities

Provide a stable, one-to-one vessel identity table and a static particulars table. The analysis schema uses `vessel_uid` for the internal vessel key and `registry_imo` for the IMO identifier. Keep the MMSI-to-vessel mapping consistent with the observation period.

`vessel_uid` is a stable integer key shared across the roster, particulars and observation tables. `registry_imo` is the nullable IMO identifier in the roster; observation tables retain the separate `imo` column. Preserve identifier values and their relationships when preparing inputs. These field names do not specify a data provider.

| Input | Required interface |
|---|---|
| Frozen roster | `mmsi`, `vessel_uid`, `registry_imo`, and the particulars consumed by the port/voyage stages, including `detailed_vessel_type` or `detailed_vessel_type_normalized` |
| Cleaning particulars | `vessel_uid`, `general_vessel_type`, `detailed_vessel_type`, `speed_service` in knots; missing service speeds follow the documented type-based fallback |
| Analysis identities | All tables use the same vessel keys; identities and the frozen roster must agree |

For example, a synthetic cleaning row can be represented as:

```csv
vessel_uid,general_vessel_type,detailed_vessel_type,speed_service
1,Cargo,Container Ship,20
```

Normalize cleaning particulars to the four column names shown above before running the cleaning stage. The complete schemas remain in `cleaning_common.py`, the unified-shard `schema.py`, and the port-call modules. The nine-category mapping remains in `vessel_group_helpers.py`. Vessel selection, speed rules and type grouping remain part of the published processing code.

Historical study controllers retain their frozen-input checksum gates. Changing column names changes file bytes, so adapting an input is not an exact replay of those frozen files. Users must configure and validate their input snapshot and its downstream processing; preserve original historical receipts rather than rewriting their hashes to accept different inputs.

## Prepared cross-year observation buffer

The public follow-up controller accepts `--buffer /path/to/prepared-february-buffer`. This replaces its former download-root argument. The buffer contains already selected, deduplicated and roster-matched observations for February 2026, partitioned into all 64 stable vessel shards. January replay uses the same schema with its January period.

Each Parquet shard contains the four position fields above plus integer `vessel_uid` and nullable integer `imo`. Shard names are `shard_0000` through `shard_0063`. Assignment is the first eight bytes of SHA-256 of the canonical MMSI string, interpreted as a big-endian integer, modulo 64. Include empty shards with the same schema.

The directory must include `fleet_presence_unique_summary.json` with:

- `status: "pass"`, `month: "2026-02"`, `probe: false`, `closure: true`, `unique_mmsi_hour: true`;
- `output_rows` and `output_vessels`, matching the actual shards;
- `buckets`: 64 records, each containing `shard`, absolute `path`, `rows` and `sha256`.

The validator reads the supplied files and checks their hashes, counts, UTC-hour bounds, coordinates, vessel-hour uniqueness, stable shard assignment and correspondence with the frozen roster. Provider response files, download plans, account information and acquisition receipts are not required by this public boundary. See `tests/test_prepared_inputs.py` for a complete synthetic fixture and rejection cases.

Cross-year reconstruction also consumes the previous validated visit, voyage, quality and weather products. Their default paths and snapshot checks remain in the stage configurations. Relocated inputs require matching manifests with the new paths; preserve original historical receipts unchanged.

## Anchorages, routing and weather

Supply the recorded anchorage/city reference tables, ocean-mask and channel references, ERA5 significant-wave-height files, IBTrACS snapshot and weather manifests before processing. Units, coordinate conventions and time windows remain specified by the stage schemas and weather configuration. The current product uses 2025 departures with observation before 1 March 2026; the weather configuration includes the required cyclone interpolation margin.

The ocean-mask utility can convert a locally supplied `earth_mask_01m_p.grd`; it raises an error if that file is missing. It does not fetch the file automatically. Frozen mask hashes and geometry checks are unchanged.

The monthly exporter can instead start from an external full-precision monthly aggregate, as described in [REPRODUCTION.md](REPRODUCTION.md). That narrower entry point performs schema validation and publication rounding, and does not reconstruct the underlying voyages.
