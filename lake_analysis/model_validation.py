"""
Model Validation and Comparison Module

Validates the exponential decay model against alternative functional forms
and provides posterior predictive checks for Bayesian models.

**Scientific Problem:**
The analysis assumes exponential decay: D(t) = D₀ × exp(-kt)
But this assumption is not tested! Alternative models might fit better:
- Linear decay: D(t) = D₀ - kt
- Power law decay: D(t) = D₀ × t^(-α)
- Logistic decay: D(t) = K / (1 + exp(r(t - t₀)))

**Solution:**
1. Fit multiple candidate models
2. Compare using AIC/BIC (penalized likelihood)
3. Perform posterior predictive checks (for Bayesian models)
4. Report model selection uncertainty

Author: morrismc
Date: 2026-01-21
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, optimize
from typing import Dict, List, Tuple, Optional, Any, Callable
import warnings

try:
    from .config import OUTPUT_DIR, ensure_output_dir
except ImportError:
    from config import OUTPUT_DIR, ensure_output_dir


def fit_exponential_decay(ages: np.ndarray, densities: np.ndarray) -> Dict[str, Any]:
    """
    Fit exponential decay model: D(t) = D0 * exp(-k * t)

    Parameters
    ----------
    ages : ndarray
        Landscape ages (ka)
    densities : ndarray
        Lake densities (lakes/1000 km²)

    Returns
    -------
    dict
        Model parameters, fit statistics, and predictions
    """
    def model(t, D0, k):
        return D0 * np.exp(-k * t)

    # Initial guess
    D0_init = np.max(densities)
    k_init = 0.001  # Half-life ~700 ka

    try:
        popt, pcov = optimize.curve_fit(
            model, ages, densities,
            p0=[D0_init, k_init],
            bounds=([0, 0], [np.inf, np.inf]),
            maxfev=10000
        )

        D0_fit, k_fit = popt
        predictions = model(ages, D0_fit, k_fit)
        residuals = densities - predictions

        # Calculate R²
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((densities - np.mean(densities)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        # Calculate AIC/BIC
        n = len(ages)
        k_params = 2  # D0, k
        log_likelihood = -0.5 * n * (np.log(2 * np.pi) + np.log(ss_res / n) + 1)
        aic = 2 * k_params - 2 * log_likelihood
        bic = k_params * np.log(n) - 2 * log_likelihood

        # Half-life
        halflife = np.log(2) / k_fit if k_fit > 0 else np.inf

        return {
            'model': 'exponential',
            'D0': D0_fit,
            'k': k_fit,
            'halflife_ka': halflife,
            'params': popt,
            'cov': pcov,
            'predictions': predictions,
            'residuals': residuals,
            'r_squared': r_squared,
            'log_likelihood': log_likelihood,
            'AIC': aic,
            'BIC': bic,
            'n_params': k_params,
            'success': True
        }

    except Exception as e:
        return {
            'model': 'exponential',
            'success': False,
            'error': str(e)
        }


def fit_linear_decay(ages: np.ndarray, densities: np.ndarray) -> Dict[str, Any]:
    """
    Fit linear decay model: D(t) = D0 - k * t

    Parameters
    ----------
    ages : ndarray
        Landscape ages (ka)
    densities : ndarray
        Lake densities (lakes/1000 km²)

    Returns
    -------
    dict
        Model parameters and fit statistics
    """
    def model(t, D0, k):
        return np.maximum(D0 - k * t, 0)  # Densities can't go negative

    try:
        popt, pcov = optimize.curve_fit(
            model, ages, densities,
            p0=[np.max(densities), 0.01],
            bounds=([0, 0], [np.inf, np.inf])
        )

        D0_fit, k_fit = popt
        predictions = model(ages, D0_fit, k_fit)
        residuals = densities - predictions

        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((densities - np.mean(densities)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        n = len(ages)
        k_params = 2
        log_likelihood = -0.5 * n * (np.log(2 * np.pi) + np.log(ss_res / n) + 1)
        aic = 2 * k_params - 2 * log_likelihood
        bic = k_params * np.log(n) - 2 * log_likelihood

        # Time to zero density (extinction time)
        extinction_time = D0_fit / k_fit if k_fit > 0 else np.inf

        return {
            'model': 'linear',
            'D0': D0_fit,
            'k': k_fit,
            'extinction_time_ka': extinction_time,
            'params': popt,
            'cov': pcov,
            'predictions': predictions,
            'residuals': residuals,
            'r_squared': r_squared,
            'log_likelihood': log_likelihood,
            'AIC': aic,
            'BIC': bic,
            'n_params': k_params,
            'success': True
        }

    except Exception as e:
        return {
            'model': 'linear',
            'success': False,
            'error': str(e)
        }


def fit_power_law_decay(ages: np.ndarray, densities: np.ndarray) -> Dict[str, Any]:
    """
    Fit power law decay model: D(t) = D0 * t^(-alpha)

    Parameters
    ----------
    ages : ndarray
        Landscape ages (ka)
    densities : ndarray
        Lake densities (lakes/1000 km²)

    Returns
    -------
    dict
        Model parameters and fit statistics
    """
    # Avoid t=0 which is undefined for power law
    valid = ages > 0
    ages_valid = ages[valid]
    densities_valid = densities[valid]

    def model(t, D0, alpha):
        return D0 * t ** (-alpha)

    try:
        popt, pcov = optimize.curve_fit(
            model, ages_valid, densities_valid,
            p0=[np.max(densities_valid), 0.5],
            bounds=([0, 0], [np.inf, 2])  # alpha typically < 2
        )

        D0_fit, alpha_fit = popt
        predictions_valid = model(ages_valid, D0_fit, alpha_fit)
        residuals_valid = densities_valid - predictions_valid

        # Expand to full array
        predictions = np.zeros_like(densities)
        predictions[valid] = predictions_valid
        residuals = np.zeros_like(densities)
        residuals[valid] = residuals_valid

        ss_res = np.sum(residuals_valid ** 2)
        ss_tot = np.sum((densities_valid - np.mean(densities_valid)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        n = len(ages_valid)
        k_params = 2
        log_likelihood = -0.5 * n * (np.log(2 * np.pi) + np.log(ss_res / n) + 1)
        aic = 2 * k_params - 2 * log_likelihood
        bic = k_params * np.log(n) - 2 * log_likelihood

        return {
            'model': 'power_law',
            'D0': D0_fit,
            'alpha': alpha_fit,
            'params': popt,
            'cov': pcov,
            'predictions': predictions,
            'residuals': residuals,
            'r_squared': r_squared,
            'log_likelihood': log_likelihood,
            'AIC': aic,
            'BIC': bic,
            'n_params': k_params,
            'success': True
        }

    except Exception as e:
        return {
            'model': 'power_law',
            'success': False,
            'error': str(e)
        }


def compare_decay_models(
    ages: np.ndarray,
    densities: np.ndarray,
    models: List[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Compare alternative decay models and select best fit.

    Parameters
    ----------
    ages : ndarray
        Landscape ages (ka)
    densities : ndarray
        Lake densities (lakes/1000 km²)
    models : list of str, optional
        Models to fit: ['exponential', 'linear', 'power_law']
        Default: all three
    verbose : bool
        Print comparison table

    Returns
    -------
    dict
        Comparison results including:
        - 'models': Dict of fitted model results
        - 'best_model': Name of best model by AIC
        - 'delta_aic': AIC differences from best
        - 'akaike_weights': Model probabilities

    Examples
    --------
    >>> comparison = compare_decay_models(ages, densities)
    >>> print(f"Best model: {comparison['best_model']}")
    >>> print(f"AIC weight: {comparison['akaike_weights'][comparison['best_model']]:.3f}")

    Notes
    -----
    Model selection uses Akaike Information Criterion (AIC).
    Lower AIC indicates better fit penalized for model complexity.
    ΔAIC < 2: Models are essentially equivalent
    ΔAIC 2-10: Best model has substantial support
    ΔAIC > 10: Best model strongly preferred

    References
    ----------
    Burnham, K. P., & Anderson, D. R. (2002). Model Selection and
    Multimodel Inference (2nd ed.). Springer.
    """
    if models is None:
        models = ['exponential', 'linear', 'power_law']

    results = {}

    # Fit each model
    for model_name in models:
        if model_name == 'exponential':
            results[model_name] = fit_exponential_decay(ages, densities)
        elif model_name == 'linear':
            results[model_name] = fit_linear_decay(ages, densities)
        elif model_name == 'power_law':
            results[model_name] = fit_power_law_decay(ages, densities)
        else:
            warnings.warn(f"Unknown model: {model_name}")

    # Extract successful models
    successful = {k: v for k, v in results.items() if v.get('success', False)}

    if len(successful) == 0:
        return {'error': 'No models converged', 'models': results}

    # Find best model (lowest AIC)
    aics = {k: v['AIC'] for k, v in successful.items()}
    best_model = min(aics, key=aics.get)
    best_aic = aics[best_model]

    # Calculate ΔAIC and Akaike weights
    delta_aic = {k: v - best_aic for k, v in aics.items()}
    akaike_weights = {}
    weight_sum = sum(np.exp(-0.5 * delta) for delta in delta_aic.values())

    for k, delta in delta_aic.items():
        akaike_weights[k] = np.exp(-0.5 * delta) / weight_sum

    comparison = {
        'models': results,
        'best_model': best_model,
        'delta_aic': delta_aic,
        'akaike_weights': akaike_weights,
        'recommendation': _get_model_recommendation(delta_aic, akaike_weights)
    }

    if verbose:
        print("\n" + "=" * 80)
        print("DECAY MODEL COMPARISON")
        print("=" * 80)
        print(f"\nFitting {len(successful)} models to {len(ages)} data points\n")
        print(f"{'Model':<15} {'R²':<10} {'AIC':<12} {'ΔAIC':<10} {'Weight':<10} {'Status':<15}")
        print("-" * 80)

        for model_name in models:
            if model_name in successful:
                r = successful[model_name]
                status = "✓" if model_name == best_model else ""
                print(f"{model_name:<15} {r['r_squared']:<10.3f} {r['AIC']:<12.2f} "
                      f"{delta_aic[model_name]:<10.2f} {akaike_weights[model_name]:<10.3f} {status:<15}")
            else:
                print(f"{model_name:<15} {'FAILED':<10} {'-':<12} {'-':<10} {'-':<10} {'✗':<15}")

        print("-" * 80)
        print(f"\nBest model: {best_model} (AIC = {best_aic:.2f})")
        print(f"Model weight: {akaike_weights[best_model]:.3f}")
        print(f"\n{comparison['recommendation']}")

        # Show best model parameters
        best = successful[best_model]
        print(f"\nBest-fit parameters ({best_model}):")
        if best_model == 'exponential':
            print(f"  D₀ = {best['D0']:.2f} lakes/1000 km²")
            print(f"  k = {best['k']:.6f} ka⁻¹")
            print(f"  Half-life = {best['halflife_ka']:.0f} ka")
        elif best_model == 'linear':
            print(f"  D₀ = {best['D0']:.2f} lakes/1000 km²")
            print(f"  k = {best['k']:.4f} lakes/1000 km²/ka")
            print(f"  Extinction time = {best['extinction_time_ka']:.0f} ka")
        elif best_model == 'power_law':
            print(f"  D₀ = {best['D0']:.2f}")
            print(f"  α = {best['alpha']:.3f}")

    return comparison


def _get_model_recommendation(delta_aic: Dict[str, float], weights: Dict[str, float]) -> str:
    """Generate model selection recommendation."""
    best_model = min(delta_aic, key=delta_aic.get)
    best_weight = weights[best_model]

    # Count competing models (ΔAIC < 2)
    competing = [k for k, v in delta_aic.items() if v < 2]

    if len(competing) == 1:
        if best_weight > 0.9:
            return f"STRONG SUPPORT for {best_model} (weight = {best_weight:.3f})"
        else:
            return f"MODERATE SUPPORT for {best_model} (weight = {best_weight:.3f})"
    else:
        return (f"MODEL UNCERTAINTY: {len(competing)} models within ΔAIC < 2. "
                f"Consider multimodel inference.")


def plot_model_comparison(
    ages: np.ndarray,
    densities: np.ndarray,
    comparison_results: Dict,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (14, 5)
) -> None:
    """
    Visualize model comparison results.

    Parameters
    ----------
    ages : ndarray
        Landscape ages
    densities : ndarray
        Lake densities
    comparison_results : dict
        Output from compare_decay_models()
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    models_fitted = {k: v for k, v in comparison_results['models'].items()
                     if v.get('success', False)}

    colors = {
        'exponential': '#E63946',
        'linear': '#457B9D',
        'power_law': '#2A9D8F'
    }

    # Panel A: Model fits
    ages_smooth = np.linspace(ages.min(), ages.max(), 100)

    for model_name, result in models_fitted.items():
        if model_name == 'exponential':
            D0, k = result['D0'], result['k']
            predictions_smooth = D0 * np.exp(-k * ages_smooth)
        elif model_name == 'linear':
            D0, k = result['D0'], result['k']
            predictions_smooth = np.maximum(D0 - k * ages_smooth, 0)
        elif model_name == 'power_law':
            D0, alpha = result['D0'], result['alpha']
            predictions_smooth = D0 * ages_smooth ** (-alpha)
        else:
            continue

        label = f"{model_name.replace('_', ' ').title()} (R²={result['r_squared']:.3f})"
        axes[0].plot(ages_smooth, predictions_smooth, '-', linewidth=2,
                    color=colors.get(model_name, 'gray'), label=label, alpha=0.8)

    # Data points
    axes[0].plot(ages, densities, 'ko', markersize=10, label='Observed', zorder=10)
    axes[0].set_xlabel('Landscape Age (ka)', fontsize=12)
    axes[0].set_ylabel('Lake Density (lakes/1000 km²)', fontsize=12)
    axes[0].set_title('A. Model Fits', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    # Panel B: AIC comparison
    model_names = list(models_fitted.keys())
    delta_aics = [comparison_results['delta_aic'][m] for m in model_names]
    colors_bar = [colors.get(m, 'gray') for m in model_names]

    axes[1].barh(model_names, delta_aics, color=colors_bar, alpha=0.7)
    axes[1].axvline(0, color='black', linewidth=2, linestyle='--')
    axes[1].axvline(2, color='red', linewidth=1, linestyle=':', label='ΔAIC = 2')
    axes[1].set_xlabel('ΔAIC (relative to best)', fontsize=12)
    axes[1].set_title('B. Model Selection (AIC)', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(axis='x', alpha=0.3)

    # Panel C: Akaike weights
    weights = [comparison_results['akaike_weights'][m] for m in model_names]

    axes[2].barh(model_names, weights, color=colors_bar, alpha=0.7)
    axes[2].set_xlabel('Akaike Weight', fontsize=12)
    axes[2].set_xlim([0, 1])
    axes[2].set_title('C. Model Probabilities', fontsize=13, fontweight='bold')
    axes[2].grid(axis='x', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison figure saved: {save_path}")
    else:
        ensure_output_dir()
        plt.savefig(f"{OUTPUT_DIR}/model_comparison.png", dpi=300, bbox_inches='tight')
        print(f"Model comparison figure saved: {OUTPUT_DIR}/model_comparison.png")

    plt.show()


# Main execution
if __name__ == "__main__":
    print("Model Validation Module")
    print("Use to test exponential decay assumption against alternatives")
