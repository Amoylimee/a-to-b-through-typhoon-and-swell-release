"""Same-domain exposure comparisons, separate from production estimators."""
from __future__ import annotations
import numpy as np
import pandas as pd


def exposure_comparison(frame: pd.DataFrame, reference: str, end_to_end: bool) -> dict:
    """M = fixed known hours; Q = hidden hours with computed bounds.

    reference='gfw' is a conditional implementation check. For 'reference_ais', the
    conditional comparison substitutes reference weather only at Q; end-to-end
    also substitutes it at M. Missing support is removed symmetrically.
    """
    q = frame.is_hidden.to_numpy(bool)
    ref_wave = frame[f'{reference}_swh'].to_numpy(float)
    gfw_wave = frame.gfw_swh.to_numpy(float)
    wave_support = np.where(q, frame.wave_status.eq('matched'), np.isfinite(gfw_wave))
    compare_wave = wave_support & np.isfinite(ref_wave)
    lo = np.where(q, frame.wave_lo_m.to_numpy(float), gfw_wave)
    hi = np.where(q, frame.wave_hi_m.to_numpy(float), gfw_wave)
    target = ref_wave if end_to_end else np.where(q, ref_wave, gfw_wave)
    n = int(compare_wave.sum())
    result = {'reference': reference, 'comparison': 'end_to_end' if end_to_end else 'fixed_known',
              'target_hours': len(frame), 'hidden_hours': int(q.sum()),
              'wave_bound_supported_hours': int(wave_support.sum()),
              'wave_unsupported_hours': int((~wave_support).sum()),
              'wave_reference_missing_on_supported_hours': int((wave_support & ~np.isfinite(ref_wave)).sum()),
              'wave_compared_hours': n, 'wave_compared_hidden_hours': int((compare_wave & q).sum()),
              'wave_lower_integral': float(lo[compare_wave].sum()),
              'wave_upper_integral': float(hi[compare_wave].sum()),
              'wave_reference_integral': float(target[compare_wave].sum())}
    for label, values in [('lower',lo),('upper',hi),('reference',target)]:
        result[f'wave_{label}_m'] = float(values[compare_wave].mean()) if n else np.nan
    result['wave_inside'] = (bool(result['wave_lower_m']-1e-6 <= result['wave_reference_m'] <= result['wave_upper_m']+1e-6) if n and result['wave_compared_hidden_hours'] else None)
    for radius in (300,500):
        ref = frame[f'{reference}_tc{radius}'].to_numpy(float)
        known = frame[f'gfw_tc{radius}'].to_numpy(float)
        valid = np.isfinite(ref) & (q | np.isfinite(known))
        lower = np.where(q,0,known)
        upper = np.where(q,frame[f'tc{radius}_possible'].to_numpy(float),known)
        target = ref if end_to_end else np.where(q,ref,known)
        count = int(valid.sum())
        result[f'tc{radius}_compared_hours'] = count
        result[f'tc{radius}_compared_hidden_hours'] = int((valid&q).sum())
        result[f'tc{radius}_reference_missing_hours'] = int((~np.isfinite(ref)).sum())
        for label, values in [('lower',lower),('upper',upper),('reference',target)]:
            result[f'tc{radius}_{label}_hours'] = float(values[valid].sum())
            result[f'tc{radius}_{label}_pct'] = float(100*values[valid].mean()) if count else np.nan
        result[f'tc{radius}_inside'] = (bool(result[f'tc{radius}_lower_hours']-1e-8 <= result[f'tc{radius}_reference_hours'] <= result[f'tc{radius}_upper_hours']+1e-8) if count and result[f'tc{radius}_compared_hidden_hours'] else None)
    return result
