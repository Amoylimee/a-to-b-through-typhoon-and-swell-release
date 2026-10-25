#!/usr/bin/env python3
"""Compare frozen IBTrACS snapshots without adopting either or inferring missing storms."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import pandas as pd
from weather_core.tropical_cyclone import load_ibtracs_snapshot


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def describe(path):
    digest=sha256(path)
    fixes,loader=load_ibtracs_snapshot(path,expected_sha256=digest)
    f=fixes[fixes.fix_time_utc.dt.year.eq(2025)].copy()
    gap=f.groupby(['SID','retained_phase']).fix_time_utc.diff().dt.total_seconds()/3600
    columns=['SID','ISO_TIME','USA_ATCF_ID','TRACK_TYPE','NATURE','LAT','LON','NAME','BASIN']
    raw=pd.read_csv(path,skiprows=[1],usecols=columns,dtype=str,keep_default_na=False)
    for c in raw:raw[c]=raw[c].str.strip()
    span=[raw.ISO_TIME.min(),raw.ISO_TIME.max()]
    raw=raw[raw.ISO_TIME.str.startswith('2025')].copy()
    result={'path':str(path.resolve()),'sha256':digest,'source_time_span':span,
            'calendar_2025_within_source_span':span[0]<='2025-01-01 00:00:00' and span[1]>='2026-01-01 00:00:00',
            'raw_2025_rows':len(raw),'raw_2025_sids':raw.SID.nunique(),
            'retained_2025_rows':len(f),'retained_2025_sids':f.SID.nunique(),
            'retained_track_types':f.track_type.value_counts().to_dict(),
            'retained_invalid_centres':int((~f.center_valid).sum()),
            'within_retained_phase_gaps_over_3h':int(gap.gt(3).sum()),
            'retained_month_counts':f.groupby(f.fix_time_utc.dt.month).size().to_dict(),
            'provisional_records_present':bool(f.track_type.ne('main').any()),
            'best_track_final':False if f.track_type.ne('main').any() else None,
            'interpretation':'Calendar coverage and internal continuity are checked; this does not prove exhaustive detection or final best-track quality.'}
    return raw,result,loader


def run(old,new,expected_new_sha,output):
    if output.exists():raise FileExistsError(output)
    if sha256(new)!=expected_new_sha:raise ValueError('new snapshot hash changed')
    a,aa,al=describe(old);b,bb,bl=describe(new)
    ai=set(a.USA_ATCF_ID)-{''};bi=set(b.USA_ATCF_ID)-{''}
    cross={'old_atcf_ids':len(ai),'new_atcf_ids':len(bi),'new_only_atcf_ids':sorted(bi-ai),'old_only_atcf_ids':sorted(ai-bi)}
    keys=['USA_ATCF_ID','ISO_TIME']
    a=a[a.USA_ATCF_ID.ne('')];b=b[b.USA_ATCF_ID.ne('')]
    ad=a.duplicated(keys,keep=False);bd=b.duplicated(keys,keep=False)
    cross.update(old_ambiguous_key_rows=int(ad.sum()),new_ambiguous_key_rows=int(bd.sum()))
    common=a.loc[~ad].merge(b.loc[~bd],on=keys,suffixes=('_old','_new'),validate='one_to_one')
    cross['common_unique_atcf_time_rows']=len(common)
    cross['changed_fields_on_common_unique_rows']={k:int(common[k+'_old'].ne(common[k+'_new']).sum()) for k in ['SID','LAT','LON','NATURE','TRACK_TYPE']}
    result={'state':'completed_read_only_audit','year':2025,'old':aa,'new':bb,'atcf_crosscheck':cross,
            'source_substitution_performed':False,'documentation':'https://www.ncei.noaa.gov/sites/default/files/2025-04/IBTrACS_version4r01_Technical_Details.pdf'}
    output.mkdir(parents=True)
    (output/'snapshot_comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (output/'old_loader_audit.json').write_text(json.dumps(al,indent=2)+'\n')
    (output/'new_loader_audit.json').write_text(json.dumps(bl,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old',type=Path,required=True);p.add_argument('--new',type=Path,required=True)
    p.add_argument('--new-sha256',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.old,a.new,a.new_sha256,a.output)
