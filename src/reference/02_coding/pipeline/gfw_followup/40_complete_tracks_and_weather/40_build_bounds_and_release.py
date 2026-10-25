"""Extend the accepted unknown-position bounds and E008 monthly candidate tables."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import pandas as pd
import pyarrow as pa

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import weather_extension as weather
core=weather.core;CODING=weather.CODING;PIPE=weather.PIPE
QUALITY=CODING/'.runs/2026-10-21_2324_E009-quality-base'
ASSETS=CODING/'.runs/2026-10-21_2330_E009-weather-assets'
OLD=CODING/'.runs/2026-10-11_1907_E007-one-cell-release'
MONTHLY=CODING/'pipeline/gfw_monthly_release'
PUBLIC_OLD=CODING/'.runs/2026-10-20_1012_E008-china-merged-release'
ORIGINAL_PUBLIC=PUBLIC_OLD
BASELINE_RELEASE=None
BASELINE_QUALITY=None
sys.path.insert(0,str(MONTHLY))
import monthly_release as monthly
EXPORT=core.module('e009_public_export',MONTHLY/'20_finalize_public_release/10_merge_china_regions.py')
from wave_support_bounds import install


def port_geometry(panel,cities):
    ports=panel[['candidate_id','source_shard','departure_time','arrival_time',*core.KEY]].copy()
    for role,city in [('departure','origin_city'),('arrival','dest_city')]:
        table=cities[['city_id','port_lat','port_lon']].rename(columns={'city_id':city,'port_lat':role+'_port_lat','port_lon':role+'_port_lon'})
        ports=ports.merge(table,on=city,how='left',validate='many_to_one')
    coords=['departure_port_lat','departure_port_lon','arrival_port_lat','arrival_port_lon']
    ports['port_endpoints_connected']=np.isfinite(ports[coords]).all(axis=1)
    return ports


def prepare(root,run):
    receipt=json.loads((run/'execution.json').read_text())
    if receipt['status']!='pass' or receipt['probe']: raise ValueError('full tracks/weather must pass')
    weather_record=receipt['summary']['output']
    if core.record(Path(weather_record['path']))!=weather_record: raise ValueError('weather facts changed')
    quality=pd.read_parquet(QUALITY/'retained_panel.parquet')
    facts=pd.read_parquet(weather_record['path'])
    columns=['candidate_id','mmsi',core.SHIP,'source_shard','departure_time','arrival_time',*core.KEY,'old_candidate_id']
    if 'e007_candidate_id' in quality:columns.append('e007_candidate_id')
    panel=quality[columns].merge(facts,on='candidate_id',how='outer',validate='one_to_one',indicator=True)
    if not panel._merge.eq('both').all(): raise ValueError('quality/weather identity differs')
    panel=panel.drop(columns='_merge')
    cities=pd.read_csv(QUALITY/'input_view/results/support/city_snap_table_2025.csv')
    ports=port_geometry(panel,cities)
    coords=['departure_port_lat','departure_port_lon','arrival_port_lat','arrival_port_lon']
    ports['port_endpoints_connected']=np.isfinite(ports[coords]).all(axis=1)
    # All reusable bound geometry needs the identical physical UTC endpoints
    # and port coordinates, as well as the already verified weather positions.
    if BASELINE_RELEASE is not None:
        old_ports=port_geometry(pd.read_parquet(BASELINE_RELEASE/'preparation/all_voyage_panel.parquet'),
                               pd.read_csv(BASELINE_QUALITY/'input_view/results/support/city_snap_table_2025.csv'))
    else:old_ports=pd.read_parquet(OLD/'preparation/port_endpoints.parquet')
    old_keys=quality.loc[quality.old_candidate_id.notna(),['candidate_id','old_candidate_id']]
    mapped=old_keys.merge(old_ports.rename(columns={'candidate_id':'old_candidate_id'}),on='old_candidate_id',validate='one_to_one')
    actual=ports.set_index('candidate_id').reindex(mapped.candidate_id)
    for col in ['departure_time','arrival_time',*coords,'port_endpoints_connected']:
        a=actual[col].reset_index(drop=True);b=mapped[col].reset_index(drop=True)
        if not (a.eq(b)|(a.isna()&b.isna())).all(): raise ValueError('old bound geometry changed: '+col)
    folder=root/'preparation';folder.mkdir()
    panel.to_parquet(folder/'all_voyage_panel.parquet',index=False)
    ports.to_parquet(folder/'all_port_endpoints.parquet',index=False)
    new=panel.loc[panel.old_candidate_id.isna()].copy()
    new.to_parquet(folder/'voyage_panel.parquet',index=False)
    ports.loc[ports.candidate_id.isin(new.candidate_id)].to_parquet(folder/'port_endpoints.parquet',index=False)
    (folder/'config').mkdir();(folder/'config/2025_ten_minute_weather.json').symlink_to(ASSETS/'weather_config.json')
    core.write_json(folder/'preparation.json',{'status':'pass','new_voyages':len(new),'new_unknown_hours':float(new.unresolved_track_hours.sum()),
                    'old_port_geometry_exact':True,'weather':weather_record,'quality':core.record(QUALITY/'execution.json'),
                    'new_panel':core.record(folder/'voyage_panel.parquet'),'ports':core.record(folder/'port_endpoints.parquet')})
    return panel


def stage_for(run,prepared):
    stage=core.module('e009_bounds',PIPE/'210_bound_gfw_missing_weather.py')
    stage.WEATHER=prepared
    original=stage.module
    def loader(name,path):
        kernels=original(name,path)
        if Path(path).name=='60_compute_tolerant_bounds.py':
            install(kernels)
            load=kernels.PROBE.load_ibtracs_snapshot
            def extended(*args,**kwargs):
                if kwargs.get('time_end_exclusive')!='2026-01-02T00:00:00Z': raise ValueError('unexpected bound calendar')
                kwargs['time_end_exclusive']=weather.TC_END;return load(*args,**kwargs)
            kernels.PROBE.load_ibtracs_snapshot=extended
            def assets(*,year,month):
                key=f'{year:04d}-{month:02d}'
                if key not in weather.MONTHS: raise ValueError('out-of-window bound weather')
                return [ASSETS/'waves'/f'swh_{key}.nc']
            kernels.PROBE.resolve_monthly_era5_assets=assets
        return kernels
    stage.module=loader
    def read(shard,*,candidates=None):
        paths=sorted((run/'weather'/shard).glob('weather_*.parquet'))
        if not paths: raise ValueError('missing new-voyage weather shard')
        filters=[('candidate_id','in',candidates)] if candidates is not None else None
        return pd.concat([pd.read_parquet(p,filters=filters) for p in paths],ignore_index=True)
    stage.read_weather=read
    return stage


def bound_worker(shard,root,run,probe):
    pa.set_cpu_count(1)
    stage=stage_for(run,root/'preparation')
    # Same 0.5 h GFW and 1 h port tolerances and 117.94 km wave support.
    result=stage.run_shard(shard,str(root/'preparation'),str(root/('bounds_probe' if probe else 'bounds')),3 if probe else None,.5,probe)
    f=pd.read_parquet(result['output']['path'])
    if len(f):
        for prefix in ['main','zero']:
            if (f[prefix+'_wave_counted_hours']>f.unknown_hours+1e-8).any(): raise ValueError('wave bound exceeds unknown time')
            if (f[prefix+'_wave_min_integral']>f[prefix+'_wave_max_integral']+1e-8).any(): raise ValueError('wave bound inverted')
            if (f[prefix+'_tc300_possible_hours']>f[prefix+'_tc500_possible_hours']+1e-8).any(): raise ValueError('cyclone bounds inverted')
            if (f[prefix+'_tc500_possible_hours']>f.unknown_hours+1e-8).any(): raise ValueError('cyclone bound exceeds unknown time')
    return result


def export(root,panel,results):
    config=json.loads((MONTHLY/'config/authorization.json').read_text())
    if BASELINE_RELEASE is not None:
        path=BASELINE_RELEASE/'private/voyage_sufficient_statistics.parquet'
        old=pd.read_parquet(path);old_receipts={str(path):core.record(path)}
    else:old,old_receipts=monthly.load_source(OLD,config)
    old_keys=panel.loc[panel.old_candidate_id.notna(),['candidate_id','old_candidate_id']]
    kept=old_keys.merge(old.rename(columns={'candidate_id':'old_candidate_id'}),on='old_candidate_id',validate='one_to_one').drop(columns='old_candidate_id')
    added=panel.loc[panel.old_candidate_id.isna()].copy()
    parts=[]
    for r in results:
        if core.record(Path(r['output']['path']))!=r['output']: raise ValueError('bound result changed')
        parts.append(pd.read_parquet(r['output']['path']))
    bound=pd.concat(parts,ignore_index=True).fillna(0)
    if bound.candidate_id.duplicated().any(): raise ValueError('duplicate unknown voyage')
    added=added.merge(bound,on='candidate_id',how='left',validate='one_to_one')
    columns=[c for c in bound if c!='candidate_id'];added[columns]=added[columns].fillna(0)
    if not np.allclose(added.unknown_hours,added.unresolved_track_hours,rtol=0,atol=1e-8): raise ValueError('new unknown hours do not close')
    for side in ['origin','dest']: added[side+'_country']=added[side+'_city'].str[:3]
    added['departure_month']=added.departure_time.dt.strftime('%Y-%m')
    combined=pd.concat([kept,added[kept.columns]],ignore_index=True)
    if len(combined)!=len(panel) or combined.candidate_id.duplicated().any(): raise ValueError('release voyage set mismatch')
    if combined[monthly.SUMS].isna().any().any() or not np.isfinite(combined[monthly.SUMS].to_numpy(float)).all(): raise ValueError('invalid sufficient statistics')
    private=root/'private';private.mkdir(mode=0o700)
    combined.to_parquet(private/'voyage_sufficient_statistics.parquet',index=False)
    mapping=pd.read_csv(PUBLIC_OLD/'private/city_code_map_PRIVATE.csv')
    missing=sorted((set(combined.origin_city)|set(combined.dest_city))-set(mapping.city_id))
    for city in missing:
        country=EXPORT.REGIONS.get(city[:3],city[:3]);codes=mapping.loc[mapping.city_code.str.startswith(country+'_'),'city_code']
        number=int(codes.str[-4:].astype(int).max())+1 if len(codes) else 1
        mapping=pd.concat([mapping,pd.DataFrame([{'city_id':city,'city_code':f'{country}_c{number:04d}'}])],ignore_index=True)
    if mapping.city_id.duplicated().any() or mapping.city_code.duplicated().any(): raise ValueError('city map collision')
    mapping.to_csv(private/'city_code_map_PRIVATE.csv',index=False)
    lookup=mapping.set_index('city_id').city_code
    for frame in [kept,combined]:
        for side in ['origin','dest']:
            frame[side+'_country']=frame[side+'_country'].replace(EXPORT.REGIONS)
            frame[side+'_city_code']=frame[side+'_city'].map(lookup)
    public=root/'public';public.mkdir()
    checks={};tables={};stats={}
    probe_ids=set(panel.loc[panel.source_shard.isin(['shard_0032','shard_0057']),'candidate_id'])
    trial=combined.loc[combined.candidate_id.isin(probe_ids)]
    trial_country=monthly.aggregate(trial,monthly.COUNTRY_KEY)
    trial_city=monthly.aggregate(trial,monthly.CODED_KEY)
    public_fields=pd.read_csv(PUBLIC_OLD/'public'/EXPORT.TABLES['country'],nrows=0).columns.tolist()
    additive=[c for c in public_fields if c.endswith('_hours') or c in {'voyages','voyages_with_tc300','voyages_with_tc500'}]
    # Restrict to fields present in both tables; aggregation includes internal sums.
    trial_check=monthly.compare(trial_city.groupby(monthly.COUNTRY_KEY)[additive].sum().reset_index(),trial_country,monthly.COUNTRY_KEY,additive,'two-shard monthly aggregation probe',atol=0)
    if int(trial_country.voyages.sum())!=len(trial):raise ValueError('monthly probe voyage count differs')
    core.write_json(root/'monthly_probe.json',{'status':'pass','shards':['shard_0032','shard_0057'],'voyages':len(trial),'city_to_country':trial_check})
    for level,keys in [('country',monthly.COUNTRY_KEY),('city',monthly.CODED_KEY)]:
        filename=EXPORT.TABLES[level]
        fields=pd.read_csv(PUBLIC_OLD/'public'/filename,nrows=0).columns.tolist()
        original=pd.read_parquet(PUBLIC_OLD/'private'/f'{level}_full_precision.parquet')
        control=monthly.aggregate(kept,keys)
        checks[level+'_old_parity']=monthly.compare(control,original,keys,[f for f in fields if f not in keys],'old monthly control',atol=0)
        if BASELINE_RELEASE is not None:
            ids=set(panel.loc[panel.e007_candidate_id.notna(),'candidate_id'])
            original_control=monthly.aggregate(combined.loc[combined.candidate_id.isin(ids)],keys)
            original_table=pd.read_parquet(ORIGINAL_PUBLIC/'private'/f'{level}_full_precision.parquet')
            checks[level+'_e007_parity']=monthly.compare(original_control,original_table,keys,[f for f in fields if f not in keys],'original E007 population',atol=0)
        table=monthly.aggregate(combined,keys)
        expected=combined.groupby(keys,observed=True).agg(voyages=(core.SHIP,'size'),ships=(core.SHIP,'nunique')).sort_index()
        pd.testing.assert_frame_equal(expected,table.set_index(keys)[['voyages','ships']].sort_index(),check_dtype=False)
        if int(table.voyages.sum())!=len(panel): raise ValueError('published voyage count differs')
        if not np.isclose(table.voyage_hours.sum(),combined.voyage_hours.sum(),rtol=0,atol=1e-6): raise ValueError('published hours differ')
        table.to_parquet(private/f'{level}_full_precision.parquet',index=False)
        tables[level],stats[level]=EXPORT.export(table,fields,public/filename)
    additive=[c for c in tables['country'] if c.endswith('_hours') or c in {'voyages','voyages_with_tc300','voyages_with_tc500'}]
    rolled=tables['city'].groupby(monthly.COUNTRY_KEY)[additive].sum().reset_index()
    checks['city_to_country']=monthly.compare(rolled,tables['country'],monthly.COUNTRY_KEY,additive,'city to country',atol=0)
    summary={'status':'pass','voyages':len(combined),'old_voyages':len(kept),'added_voyages':len(added),
             'voyage_hours':float(combined.voyage_hours.sum()),'added_voyage_hours':float(added.voyage_hours.sum()),
             'added_unknown_hours':float(added.unknown_hours.sum()),'new_city_codes':len(missing),'tables':stats,'checks':checks,
             'old_source_sha256':old_receipts,'scientific_adoption':False,'public_upload':False,'bootstrap_computed':False}
    last_month=(pd.Timestamp(weather.OBSERVATION_END)-pd.Timedelta(days=1)).strftime('%B %Y')
    text=f'''# GFW voyage exposure: 2025 departures with {last_month} follow-up

Review candidate, not an adopted or published replacement. This package covers {len(combined):,} voyages departing during UTC 2025, including arrivals observed through {last_month}. This finite buffer does not establish complete follow-up of every 2025 departure.

The country table contains {stats['country']['records']:,} records; the coded city table contains {stats['city']['records']:,}. Each record groups departure month, directed origin/destination and vessel type. All groups with at least one retained voyage are included. The two tables use the same voyages and {combined.voyage_hours.sum():,.0f} voyage-hours.

China is coded CHN, including Hong Kong, China, Taiwan, China and Macao, China. Existing anonymous city codes are preserved; new cities receive additional codes. Raw positions, vessel identities and city-name mappings are excluded.

Methods, field names and rounding match the preceding E008 candidate. Tracks keep unresolved gaps; missing exposure is not filled with zero. Wave and cyclone ranges describe the accepted missing-position support calculation, not a guarantee of true exposure. Cyclone tracks include provisional records. Sampling confidence intervals remain omitted as previously requested.

Read 03_FIELD_GUIDE.md for field meanings. 04_MANIFEST.json records the file hashes. This local review package has not been uploaded, assigned a new DOI version, or substituted for the existing release.
'''
    (public/'00_README.md').write_text(text)
    (public/'03_FIELD_GUIDE.md').write_bytes((PUBLIC_OLD/'public/03_FIELD_GUIDE.md').read_bytes())
    core.write_json(public/'04_MANIFEST.json',{name:{k:v for k,v in core.record(public/name).items() if k!='path'} for name in sorted(EXPORT.FILES)})
    core.write_json(root/'summary.json',summary)
    return summary


def main():
    global QUALITY,ASSETS,BASELINE_RELEASE,BASELINE_QUALITY,PUBLIC_OLD
    p=argparse.ArgumentParser();p.add_argument('--weather',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,default=16);p.add_argument('--quality',type=Path,default=QUALITY)
    p.add_argument('--assets',type=Path,default=ASSETS);p.add_argument('--baseline-release',type=Path);p.add_argument('--baseline-quality',type=Path)
    a=p.parse_args();QUALITY=a.quality.resolve();ASSETS=a.assets.resolve();BASELINE_RELEASE=a.baseline_release;BASELINE_QUALITY=a.baseline_quality
    if BASELINE_RELEASE is not None:
        if BASELINE_QUALITY is None:raise ValueError('baseline quality required for old port geometry')
        PUBLIC_OLD=BASELINE_RELEASE
    weather.configure(json.loads((ASSETS/'manifest.json').read_text()).get('observation_end_exclusive','2026-02-01'))
    root=core.inside_runs(a.out);root.mkdir(parents=True,exist_ok=False)
    code=[core.record(p) for p in [Path(__file__),HERE/'weather_extension.py',PIPE/'210_bound_gfw_missing_weather.py',PIPE/'wave_support_bounds.py',MONTHLY/'monthly_release.py']]
    state={'status':'running','phase':'preparation','code':code,'weather':str(a.weather.resolve()),'results':[],'failures':[],
           'authorization':'2026-10-22 explicit February Worker instruction' if BASELINE_RELEASE is not None else '2026-10-21 recorded research instruction','scientific_adoption':False}
    save=lambda:core.write_json(root/'execution.json',state);save();start=time.monotonic()
    try:
        field_counts=[len(pd.read_csv(PUBLIC_OLD/'public'/EXPORT.TABLES[x],nrows=0).columns) for x in ['country','city']]
        if field_counts!=[41,43]:raise ValueError('public field counts differ from protected 41/43 schema')
        core.write_json(root/'required_actual.json',{'status':'pass','checks':[
            {'parameter':'observation_cutoff_UTC','Required':weather.OBSERVATION_END,'Actual':weather.OBSERVATION_END,'Status':'pass'},
            {'parameter':'bounds_kernel','Required':'accepted production kernel with wave support correction','Actual':code[2:4],'Status':'pass'},
            {'parameter':'public_fields','Required':[41,43],'Actual':field_counts,'Status':'pass'},
            {'parameter':'sampling_intervals','Required':False,'Actual':False,'Status':'pass'},
            {'parameter':'city_codes','Required':'preserve existing codes and append only','Actual':str(PUBLIC_OLD/'private/city_code_map_PRIVATE.csv'),'Status':'pass'}]})
        panel=prepare(root,a.weather.resolve());state['phase']='bounds_probe';save()
        with ProcessPoolExecutor(max_workers=2) as pool:
            state['probe']=[f.result() for f in [pool.submit(bound_worker,s,root,a.weather.resolve(),True) for s in ['shard_0032','shard_0057']]]
        state['phase']='bounds_all';save()
        shards=sorted(pd.read_parquet(root/'preparation/voyage_panel.parquet',columns=['source_shard']).source_shard.unique())
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            jobs={pool.submit(bound_worker,s,root,a.weather.resolve(),False):s for s in shards}
            for future in as_completed(jobs):
                try: state['results'].append(future.result())
                except BaseException: state['failures'].append({'shard':jobs[future],'error':traceback.format_exc()})
                save();print(json.dumps({'bounds_done':len(state['results']),'failed':len(state['failures'])}),flush=True)
        if state['failures']: raise RuntimeError('bounds shards failed')
        state['phase']='monthly_tables';save();state['summary']=export(root,panel,state['results'])
        if [core.record(Path(r['path'])) for r in code]!=code: raise ValueError('code changed during run')
        state.update(status='pass',phase='review_candidate_complete')
    except BaseException:
        state.update(status='failed',error=traceback.format_exc());raise
    finally: state['seconds']=time.monotonic()-start;save()


if __name__=='__main__':main()
