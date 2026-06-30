"""
================================================================
 ML PARAMETER ESTIMATION — Batman Transit Fitting + Uncertainty
================================================================
 For targets classified as PLANET:
   1. Initial parameters from TLS
   2. Batman model fitting with scipy.optimize
   3. Uncertainty via lmfit (covariance matrix)
   4. Dilution-corrected radius chain with error propagation
   
 Output: period ± σ, depth ± σ, duration ± σ, Rp ± σ, etc.
================================================================
"""

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import os

from scipy.optimize import minimize
from scipy.stats import median_abs_deviation

try:
    import batman
    HAS_BATMAN = True
except ImportError:
    HAS_BATMAN = False
    print("[PARAM EST] batman-package not installed. Using simplified model.")

try:
    import lmfit
    HAS_LMFIT = True
except ImportError:
    HAS_LMFIT = False
    print("[PARAM EST] lmfit not installed. Using scipy uncertainty estimates.")


# ── Config
RESULTS_DIR = './tess_pipeline_output/ml_results'
os.makedirs(RESULTS_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# TRANSIT MODEL
# ═══════════════════════════════════════════════════════════════

def transit_model_batman(t, period, t0, rp_rs, a_rs, inc, u1=0.3, u2=0.1):
    """
    Generate a batman transit model light curve.
    
    Parameters:
        t: time array
        period: orbital period (days)
        t0: mid-transit time
        rp_rs: planet-to-star radius ratio (Rp/R*)
        a_rs: scaled semi-major axis (a/R*)
        inc: inclination (degrees)
        u1, u2: quadratic limb-darkening coefficients
    """
    if not HAS_BATMAN:
        return transit_model_simple(t, period, t0, rp_rs, a_rs, inc)
    
    params = batman.TransitParams()
    params.per = period
    params.t0 = t0
    params.rp = rp_rs
    params.a = a_rs
    params.inc = inc
    params.ecc = 0.0
    params.w = 90.0
    params.limb_dark = "quadratic"
    params.u = [u1, u2]
    
    m = batman.TransitModel(params, t)
    return m.light_curve(params)


def transit_model_simple(t, period, t0, rp_rs, a_rs, inc, u1=0.3, u2=0.1):
    """
    Simplified trapezoidal transit model (fallback if batman not installed).
    """
    # Compute impact parameter
    b = a_rs * np.cos(np.radians(inc))
    
    # Transit duration (approximate)
    dur = period / np.pi * np.arcsin(
        1.0 / a_rs * np.sqrt((1 + rp_rs)**2 - b**2) / np.sin(np.radians(inc))
    ) if abs(b) < 1 + rp_rs else 0.0
    
    depth = rp_rs**2
    ingress = 0.15 * dur  # Approximate ingress fraction
    
    flux = np.ones_like(t)
    phase = ((t - t0 + 0.5 * period) % period) / period - 0.5
    
    half_dur = dur / (2 * period)
    half_ing = ingress / (2 * period)
    
    # Flat bottom
    in_transit = np.abs(phase) < (half_dur - half_ing)
    flux[in_transit] = 1.0 - depth
    
    # Ingress / egress
    in_ingress = (np.abs(phase) >= (half_dur - half_ing)) & (np.abs(phase) < half_dur)
    if in_ingress.sum() > 0:
        frac = (half_dur - np.abs(phase[in_ingress])) / max(half_ing, 1e-10)
        flux[in_ingress] = 1.0 - depth * np.clip(frac, 0, 1)
    
    return flux


def get_limb_darkening_coeffs(teff, logg=4.4):
    """
    Approximate quadratic limb-darkening coefficients for TESS bandpass
    based on stellar effective temperature.
    
    Uses simplified Claret (2017) TESS interpolation.
    """
    # Continuous approximation for TESS bandpass
    teff = np.clip(teff, 3500, 8000)
    u1 = 0.5 - ((teff - 3500) / 4500) * 0.3
    u2 = 0.25 - ((teff - 3500) / 4500) * 0.15
    return u1, u2


# ═══════════════════════════════════════════════════════════════
# PARAMETER ESTIMATION
# ═══════════════════════════════════════════════════════════════

def estimate_parameters(time, flux, flux_err, tls_result, 
                         stellar_params=None, crowdsap=1.0):
    """
    Fit a transit model to the light curve and estimate parameters
    with uncertainties.
    
    Parameters:
        time: time array (cleaned)
        flux: flux array (detrended, normalized)
        flux_err: flux error array
        tls_result: TLS results object with initial parameters
        stellar_params: dict with 'rad', 'mass', 'teff', 'logg'
        crowdsap: CROWDSAP value for dilution correction
    
    Returns:
        params_dict: estimated parameters with uncertainties
    """
    if stellar_params is None:
        stellar_params = {'rad': 1.0, 'mass': 1.0, 'teff': 5500, 'logg': 4.4}
    
    R_star = stellar_params.get('rad', 1.0)      # Solar radii
    M_star = stellar_params.get('mass', 1.0)      # Solar masses
    Teff   = stellar_params.get('teff', 5500)
    
    # ── Initial parameters from TLS
    period_init = tls_result.period
    t0_init = tls_result.T0
    depth_init = abs(1.0 - tls_result.depth)
    duration_init = tls_result.duration
    
    # Initial Rp/R* (corrected for dilution)
    corrected_depth = depth_init / max(crowdsap, 0.01)
    rp_rs_init = np.sqrt(abs(corrected_depth))
    
    # Initial a/R* from Kepler's third law
    # a/R* = (P^2 * G * M_star / (4π²))^(1/3) / R_star
    G = 6.674e-11  # m^3 kg^-1 s^-2
    M_sun = 1.989e30  # kg
    R_sun = 6.957e8   # m
    P_seconds = period_init * 86400
    
    a_meters = (G * M_star * M_sun * P_seconds**2 / (4 * np.pi**2))**(1/3)
    a_rs_init = a_meters / (R_star * R_sun)
    
    # Initial inclination from impact parameter
    # Approximate b from TLS (use transit shape)
    inc_init = 88.0  # degrees, close to edge-on
    
    # Limb darkening
    u1, u2 = get_limb_darkening_coeffs(Teff, stellar_params.get('logg', 4.4))
    
    print(f"\n  Initial parameters from TLS:")
    print(f"    Period     : {period_init:.6f} d")
    print(f"    T0         : {t0_init:.6f}")
    print(f"    Depth      : {depth_init*1e6:.1f} ppm (obs) → {corrected_depth*1e6:.1f} ppm (corr)")
    print(f"    Duration   : {duration_init*24:.2f} hrs")
    print(f"    Rp/R*      : {rp_rs_init:.6f}")
    print(f"    a/R*       : {a_rs_init:.2f}")
    print(f"    LD coeffs  : u1={u1:.2f}, u2={u2:.2f}")
    
    # ── Phase-fold for fitting
    phase = ((time - t0_init + 0.5 * period_init) % period_init) / period_init - 0.5
    sort_idx = np.argsort(phase)
    t_fold = phase[sort_idx] * period_init + t0_init  # Convert back to time-like
    f_fold = flux[sort_idx]
    e_fold = flux_err[sort_idx] if flux_err is not None else np.ones_like(f_fold) * 0.001
    
    # Clip to near-transit region for fitting efficiency
    near_transit = np.abs(phase[sort_idx]) < 0.15
    t_fit = t_fold[near_transit]
    f_fit = f_fold[near_transit]
    e_fit = e_fold[near_transit]
    
    if len(t_fit) < 20:
        print("  WARNING: Too few points near transit for fitting. Using TLS parameters.")
        return _tls_fallback_params(tls_result, stellar_params, crowdsap)
    
    # ── Fit with lmfit (if available) or scipy
    if HAS_LMFIT and HAS_BATMAN:
        result = _fit_with_lmfit(t_fit, f_fit, e_fit, period_init, t0_init,
                                  rp_rs_init, a_rs_init, inc_init, u1, u2)
    else:
        result = _fit_with_scipy(t_fit, f_fit, e_fit, period_init, t0_init,
                                  rp_rs_init, a_rs_init, inc_init, u1, u2)
    
    # ── Compute derived parameters with uncertainty propagation
    params = _compute_derived_params(result, stellar_params, crowdsap)
    
    return params


def _fit_with_lmfit(t, f, e, period, t0, rp_rs, a_rs, inc, u1, u2):
    """Fit transit model using lmfit (provides uncertainty via covariance)."""
    print("\n  Fitting with lmfit + batman...")
    
    params = lmfit.Parameters()
    params.add('rp_rs', value=rp_rs, min=0.001, max=0.3)
    params.add('a_rs', value=a_rs, min=2.0, max=200.0)
    params.add('inc', value=inc, min=70.0, max=90.0)
    params.add('t0', value=t0, vary=True)
    # Period held fixed (well-determined by TLS)
    params.add('period', value=period, vary=False)
    params.add('u1', value=u1, vary=False)
    params.add('u2', value=u2, vary=False)
    
    def residual(p):
        model = transit_model_batman(
            t, p['period'], p['t0'], p['rp_rs'], p['a_rs'], p['inc'],
            p['u1'], p['u2']
        )
        return (f - model) / e
    
    result = lmfit.minimize(residual, params, method='leastsq')
    
    # Extract results
    fit_result = {
        'rp_rs': result.params['rp_rs'].value,
        'rp_rs_err': result.params['rp_rs'].stderr or 0.0,
        'a_rs': result.params['a_rs'].value,
        'a_rs_err': result.params['a_rs'].stderr or 0.0,
        'inc': result.params['inc'].value,
        'inc_err': result.params['inc'].stderr or 0.0,
        't0': result.params['t0'].value,
        't0_err': result.params['t0'].stderr or 0.0,
        'period': period,
        'period_err': 0.0,  # Fixed
        'u1': u1, 'u2': u2,
        'chi2': result.chisqr,
        'redchi2': result.redchi,
        'bic': result.bic,
        'success': result.success,
        'method': 'lmfit',
    }
    
    print(f"    Fit {'succeeded' if result.success else 'FAILED'}: "
          f"χ²_red = {result.redchi:.4f}")
    
    return fit_result


def _fit_with_scipy(t, f, e, period, t0, rp_rs, a_rs, inc, u1, u2):
    """Fit transit model using scipy.optimize (fallback)."""
    print("\n  Fitting with scipy.optimize...")
    
    def chi2(params):
        rp_rs_p, a_rs_p, inc_p, t0_p = params
        if rp_rs_p < 0.001 or rp_rs_p > 0.3:
            return 1e10
        if a_rs_p < 2.0 or a_rs_p > 200.0:
            return 1e10
        if inc_p < 70.0 or inc_p > 90.0:
            return 1e10
        
        model = transit_model_batman(t, period, t0_p, rp_rs_p, a_rs_p, inc_p, u1, u2)
        residuals = (f - model) / e
        return np.sum(residuals**2)
    
    x0 = [rp_rs, a_rs, inc, t0]
    result = minimize(chi2, x0, method='Nelder-Mead',
                      options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
    
    # Estimate uncertainties from numerical Hessian
    from scipy.optimize import approx_fprime
    
    def neg_loglike(p):
        return chi2(p) / 2.0
    
    try:
        eps = np.array([1e-6, 0.1, 0.01, 1e-5])
        hessian = np.zeros((4, 4))
        for i in range(4):
            def grad_i(p):
                return approx_fprime(p, neg_loglike, eps[i])[i]
            hessian[i, :] = approx_fprime(result.x, grad_i, eps)
        
        cov = np.linalg.inv(hessian + np.eye(4) * 1e-10)
        errors = np.sqrt(np.abs(np.diag(cov)))
    except Exception:
        errors = np.zeros(4)
    
    fit_result = {
        'rp_rs': result.x[0],
        'rp_rs_err': errors[0],
        'a_rs': result.x[1],
        'a_rs_err': errors[1],
        'inc': result.x[2],
        'inc_err': errors[2],
        't0': result.x[3],
        't0_err': errors[3],
        'period': period,
        'period_err': 0.0,
        'u1': u1, 'u2': u2,
        'chi2': result.fun,
        'redchi2': result.fun / max(len(t) - 4, 1),
        'bic': result.fun + 4 * np.log(len(t)),
        'success': result.success,
        'method': 'scipy',
    }
    
    print(f"    Fit {'succeeded' if result.success else 'FAILED'}: "
          f"χ² = {result.fun:.4f}")
    
    return fit_result


def _tls_fallback_params(tls_result, stellar_params, crowdsap):
    """Use TLS parameters directly when fitting fails."""
    R_star = stellar_params.get('rad', 1.0)
    
    depth = abs(1.0 - tls_result.depth)
    corrected_depth = depth / max(crowdsap, 0.01)
    rp_rs = np.sqrt(abs(corrected_depth))
    Rp_earth = rp_rs * R_star * 109.076  # R_sun/R_earth ≈ 109.076
    
    return {
        'period': tls_result.period,
        'period_err': 0.0,
        'period_unit': 'days',
        't0': tls_result.T0,
        't0_err': 0.0,
        'depth_ppm': corrected_depth * 1e6,
        'depth_ppm_err': 0.0,
        'duration_hrs': tls_result.duration * 24,
        'duration_hrs_err': 0.0,
        'rp_rs': rp_rs,
        'rp_rs_err': 0.0,
        'Rp_earth': Rp_earth,
        'Rp_earth_err': 0.0,
        'inc': 90.0,
        'inc_err': 0.0,
        'a_rs': 0.0,
        'a_rs_err': 0.0,
        'b': 0.0,
        'b_err': 0.0,
        'a_AU': 0.0,
        'a_AU_err': 0.0,
        'fit_method': 'tls_fallback',
        'chi2_red': 0.0,
    }


def _compute_derived_params(fit_result, stellar_params, crowdsap):
    """
    Compute derived physical parameters with uncertainty propagation.
    
    Key chain: depth_obs → depth_corr (÷ CROWDSAP) → Rp (× R_star × 109.076)
    """
    R_star = stellar_params.get('rad', 1.0)       # Solar radii
    R_star_err = stellar_params.get('rad_err', R_star * 0.05)  # Default 5%
    M_star = stellar_params.get('mass', 1.0)
    
    rp_rs = fit_result['rp_rs']
    rp_rs_err = fit_result['rp_rs_err']
    a_rs = fit_result['a_rs']
    a_rs_err = fit_result['a_rs_err']
    inc = fit_result['inc']
    inc_err = fit_result['inc_err']
    
    # ── Transit depth (corrected for dilution)
    depth_obs = rp_rs**2
    depth_corr = depth_obs / max(crowdsap, 0.01)
    
    # CROWDSAP uncertainty (typically ±0.02-0.05)
    crowdsap_err = 0.03
    
    # depth_corr_err via error propagation:
    # d_corr = d_obs / C → σ_d = sqrt((σ_d_obs/C)² + (d_obs * σ_C / C²)²)
    depth_obs_err = 2 * rp_rs * rp_rs_err  # d(rp²)/drp = 2*rp
    depth_corr_err = np.sqrt(
        (depth_obs_err / max(crowdsap, 0.01))**2 +
        (depth_obs * crowdsap_err / max(crowdsap, 0.01)**2)**2
    )
    
    # ── Planet radius with uncertainty
    # Rp = sqrt(depth_corr) × R_star × 109.076
    rp_corr = np.sqrt(depth_corr) if depth_corr > 0 else 0
    Rp_earth = rp_corr * R_star * 109.076
    
    # Error propagation for Rp_earth
    # Rp = sqrt(d) * R * 109.076
    # σ_Rp = 109.076 * sqrt((R * σ_d / (2*sqrt(d)))² + (sqrt(d) * σ_R)²)
    if depth_corr > 0:
        term1 = (R_star * depth_corr_err / (2 * np.sqrt(depth_corr)))**2
    else:
        term1 = 0
    term2 = (np.sqrt(abs(depth_corr)) * R_star_err)**2
    Rp_earth_err = 109.076 * np.sqrt(term1 + term2)
    
    # ── Impact parameter
    b = a_rs * np.cos(np.radians(inc))
    b_err = np.sqrt(
        (np.cos(np.radians(inc)) * a_rs_err)**2 +
        (a_rs * np.sin(np.radians(inc)) * np.radians(inc_err))**2
    )
    
    # ── Transit duration (from model parameters)
    if a_rs > 0 and abs(b) < 1 + rp_rs:
        try:
            duration_days = (fit_result['period'] / np.pi) * np.arcsin(
                (1.0 / a_rs) * np.sqrt((1 + rp_rs)**2 - b**2) / np.sin(np.radians(inc))
            )
        except (ValueError, RuntimeWarning):
            duration_days = 0.0
    else:
        duration_days = 0.0
    duration_hrs = duration_days * 24
    
    # Rough duration uncertainty (10% of duration or from b/a_rs errors)
    duration_hrs_err = duration_hrs * 0.1  # Conservative estimate
    
    # ── Semi-major axis in AU
    R_sun_AU = 0.00465047  # Solar radii to AU
    a_AU = a_rs * R_star * R_sun_AU
    a_AU_err = np.sqrt(
        (R_star * R_sun_AU * a_rs_err)**2 +
        (a_rs * R_sun_AU * R_star_err)**2
    )
    
    params = {
        'period': fit_result['period'],
        'period_err': fit_result['period_err'],
        'period_unit': 'days',
        't0': fit_result['t0'],
        't0_err': fit_result['t0_err'],
        'depth_ppm': depth_corr * 1e6,
        'depth_ppm_err': depth_corr_err * 1e6,
        'duration_hrs': duration_hrs,
        'duration_hrs_err': duration_hrs_err,
        'rp_rs': rp_rs,
        'rp_rs_err': rp_rs_err,
        'Rp_earth': Rp_earth,
        'Rp_earth_err': Rp_earth_err,
        'inc': inc,
        'inc_err': inc_err,
        'a_rs': a_rs,
        'a_rs_err': a_rs_err,
        'b': b,
        'b_err': b_err,
        'a_AU': a_AU,
        'a_AU_err': a_AU_err,
        'fit_method': fit_result['method'],
        'chi2_red': fit_result['redchi2'],
    }
    
    # Print summary
    print(f"\n  ╔═══════════════════════════════════════════════╗")
    print(f"  ║  ESTIMATED TRANSIT PARAMETERS                 ║")
    print(f"  ╠═══════════════════════════════════════════════╣")
    print(f"  ║  Period      : {params['period']:.6f} ± {params['period_err']:.6f} d  ║")
    print(f"  ║  Depth       : {params['depth_ppm']:.1f} ± {params['depth_ppm_err']:.1f} ppm       ║")
    print(f"  ║  Duration    : {params['duration_hrs']:.2f} ± {params['duration_hrs_err']:.2f} hrs       ║")
    print(f"  ║  Rp/R*       : {params['rp_rs']:.6f} ± {params['rp_rs_err']:.6f}     ║")
    print(f"  ║  Rp          : {params['Rp_earth']:.2f} ± {params['Rp_earth_err']:.2f} R⊕           ║")
    print(f"  ║  Inclination : {params['inc']:.2f} ± {params['inc_err']:.2f} deg        ║")
    print(f"  ║  a/R*        : {params['a_rs']:.2f} ± {params['a_rs_err']:.2f}              ║")
    print(f"  ║  Impact (b)  : {params['b']:.4f} ± {params['b_err']:.4f}          ║")
    print(f"  ║  a (AU)      : {params['a_AU']:.5f} ± {params['a_AU_err']:.5f} AU   ║")
    print(f"  ║  Fit method  : {params['fit_method']}                  ║")
    print(f"  ║  χ²_red      : {params['chi2_red']:.4f}                       ║")
    print(f"  ╚═══════════════════════════════════════════════╝")
    
    return params


def estimate_all_candidates(candidates_df, sector=1, use_network=False):
    """
    Estimate parameters for all planet candidates in a DataFrame.
    
    Args:
        candidates_df: DataFrame with tic_id, period, T0, etc.
        sector: TESS sector
        use_network: Whether to download real data
    
    Returns:
        List of parameter dicts
    """
    from dataprepro2 import (stage1_ingest, stage2_quality_mask, stage3_detrend,
                             stage4_sigma_clip, stage5_tls, stage_crowdsap_check,
                             transit_mask)
    
    all_params = []
    
    for i, row in candidates_df.iterrows():
        tic_id = int(row['tic_id'])
        print(f"\n{'='*60}")
        print(f"  Parameter Estimation: TIC {tic_id} (Sector {sector})")
        print(f"{'='*60}")
        
        try:
            # Re-run pipeline to get clean light curve
            if not use_network:
                from dataprepro2 import make_synthetic_tess_lc
                period_val = row.get('period', 3.4)
                depth_val = row.get('depth_ppm_obs', 1800.0) / 1e6
                data, meta = make_synthetic_tess_lc(
                    tic_id=tic_id, sector=sector,
                    period=period_val, depth=depth_val
                )
                meta['flux_source'] = 'synthetic'
                source = 'synthetic'
                crowdsap_result = stage_crowdsap_check(meta)
                (time_q, flux_q, err_q, cc_q, cr_q, _, _) = stage2_quality_mask(data)
            else:
                data, meta, source = stage1_ingest(tic_id, sector, use_network)
                crowdsap_result = stage_crowdsap_check(meta)
                (time_q, flux_q, err_q, cc_q, cr_q, _, _) = stage2_quality_mask(data)
            
            flux_src = meta.get('flux_source', 'PDCSAP')
            flat, trend, _, _, _ = stage3_detrend(time_q, flux_q, flux_src)
            (time_c, flux_c, err_c, ccol_c, crow_c, _) = stage4_sigma_clip(
                time_q, flat, err_q, cc_q, cr_q, sigma_upper=5.0)
            
            if len(time_c) < 500:
                print(f"  Skipping TIC {tic_id}: insufficient data")
                continue
            
            tls_res, _, _ = stage5_tls(time_c, flux_c, err_c)
            
            # Iterative detrending
            t_mask = transit_mask(time_q, period=tls_res.period, 
                                  duration=tls_res.duration * 1.5, T0=tls_res.T0)
            adaptive_window = max(3.0 * tls_res.duration, 0.5)
            flat2, trend2, _, _, _ = stage3_detrend(
                time_q, flux_q, flux_src, transit_mask=t_mask, 
                window_override=adaptive_window)
            (time_c, flux_c, err_c, _, _, _) = stage4_sigma_clip(
                time_q, flat2, err_q, cc_q, cr_q, sigma_upper=5.0)
            tls_res, _, _ = stage5_tls(time_c, flux_c, err_c)
            
            stellar = {
                'rad': meta.get('rad', 1.0),
                'mass': meta.get('mass', 1.0),
                'teff': meta.get('TEFF', 5500),
                'logg': meta.get('LOGG', 4.4),
            }
            
            params = estimate_parameters(
                time_c, flux_c, err_c, tls_res,
                stellar_params=stellar,
                crowdsap=crowdsap_result['crowdsap']
            )
            params['tic_id'] = tic_id
            params['sector'] = sector
            all_params.append(params)
            
        except Exception as e:
            print(f"  ERROR for TIC {tic_id}: {e}")
            continue
    
    # Save results
    if all_params:
        params_df = pd.DataFrame(all_params)
        out_path = os.path.join(RESULTS_DIR, 'parameter_estimates.csv')
        params_df.to_csv(out_path, index=False)
        print(f"\n[PARAM EST] Saved {len(all_params)} parameter estimates to {out_path}")
    
    return all_params


# ═══════════════════════════════════════════════════════════════
# STANDALONE DEMO
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    from dataprepro2 import (make_synthetic_tess_lc, stage2_quality_mask, 
                             stage3_detrend, stage4_sigma_clip, stage5_tls,
                             stage_crowdsap_check, transit_mask)
    
    print("=" * 60)
    print("  PARAMETER ESTIMATION DEMO")
    print("=" * 60)
    
    # Create a synthetic planet with known parameters
    true_period = 3.4
    true_depth = 1800e-6
    true_duration = 2.1
    
    data, meta = make_synthetic_tess_lc(
        tic_id=261136679, sector=1,
        period=true_period, depth=true_depth,
        duration_hrs=true_duration, noise_ppm=600,
        crowdsap=0.95
    )
    meta['rad'] = 1.0
    meta['flux_source'] = 'PDCSAP'
    
    # Process through pipeline
    crowdsap_result = stage_crowdsap_check(meta)
    (time_q, flux_q, err_q, cc_q, cr_q, _, _) = stage2_quality_mask(data)
    flat, trend, _, _, _ = stage3_detrend(time_q, flux_q, 'PDCSAP')
    (time_c, flux_c, err_c, _, _, _) = stage4_sigma_clip(
        time_q, flat, err_q, cc_q, cr_q, sigma_upper=5.0)
    tls_res, _, _ = stage5_tls(time_c, flux_c, err_c)
    
    # Estimate parameters
    params = estimate_parameters(
        time_c, flux_c, err_c, tls_res,
        stellar_params={'rad': 1.0, 'mass': 1.0, 'teff': 5500, 'logg': 4.4},
        crowdsap=0.95
    )
    
    # Compare with truth
    print(f"\n  Ground Truth vs Estimated:")
    print(f"    Period  : {true_period:.4f} d  vs  {params['period']:.4f} d")
    print(f"    Depth   : {true_depth*1e6:.1f} ppm  vs  {params['depth_ppm']:.1f} ppm")
