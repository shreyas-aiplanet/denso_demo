"""
Data Validation and Quality Checks
Ensures training data meets minimum requirements for time-series forecasting
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
import warnings


class DataValidator:
    """
    Validates time-series data for forecasting model training
    """

    def __init__(self, min_samples=12, max_missing_pct=0.1):
        """
        Initialize the data validator

        Parameters:
        -----------
        min_samples : int
            Minimum number of data points required (default: 12 for quarterly over 3 years)
        max_missing_pct : float
            Maximum percentage of missing values allowed (default: 10%)
        """
        self.min_samples = min_samples
        self.max_missing_pct = max_missing_pct
        self.validation_results = {}
        self.warnings = []
        self.errors = []

    def validate(self, data, verbose=True):
        """
        Run comprehensive validation on time-series data

        Parameters:
        -----------
        data : array-like
            Time-series data to validate
        verbose : bool
            Whether to print validation results

        Returns:
        --------
        dict : Validation results with pass/fail status
        """
        self.warnings = []
        self.errors = []

        data = np.array(data)

        # Run all validation checks
        self._check_data_length(data)
        self._check_missing_values(data)
        self._check_data_types(data)
        self._check_negative_values(data)
        self._check_outliers(data)
        self._check_variance(data)
        self._check_stationarity(data)
        self._check_trend(data)
        self._check_seasonality(data)

        # Determine overall pass/fail
        has_errors = len(self.errors) > 0
        has_warnings = len(self.warnings) > 0

        if verbose:
            self._print_results()

        return {
            'passed': not has_errors,
            'has_warnings': has_warnings,
            'errors': self.errors,
            'warnings': self.warnings,
            'details': self.validation_results
        }

    def _check_data_length(self, data):
        """Check if data has sufficient length"""
        length = len(data)
        self.validation_results['data_length'] = length

        if length < self.min_samples:
            self.errors.append(
                f"Insufficient data: {length} samples (minimum required: {self.min_samples}). "
                f"Time-series models require adequate historical data for reliable predictions."
            )
        elif length < 30:
            self.warnings.append(
                f"Limited data: {length} samples. More historical data (30+) would improve model accuracy."
            )

    def _check_missing_values(self, data):
        """Check for missing or NaN values"""
        if pd.isna(data).any():
            missing_count = pd.isna(data).sum()
            missing_pct = missing_count / len(data)
            self.validation_results['missing_values'] = missing_count
            self.validation_results['missing_percentage'] = missing_pct * 100

            if missing_pct > self.max_missing_pct:
                self.errors.append(
                    f"Too many missing values: {missing_count} ({missing_pct*100:.1f}%). "
                    f"Maximum allowed: {self.max_missing_pct*100}%"
                )
            else:
                self.warnings.append(
                    f"Missing values detected: {missing_count} ({missing_pct*100:.1f}%). "
                    f"Consider imputation or removal."
                )
        else:
            self.validation_results['missing_values'] = 0

    def _check_data_types(self, data):
        """Check if data contains valid numeric values"""
        try:
            numeric_data = pd.to_numeric(data, errors='coerce')
            if pd.isna(numeric_data).any():
                non_numeric_count = pd.isna(numeric_data).sum()
                self.errors.append(
                    f"Non-numeric values detected: {non_numeric_count} values cannot be converted to numbers"
                )
        except Exception as e:
            self.errors.append(f"Data type validation failed: {str(e)}")

    def _check_negative_values(self, data):
        """Check for negative values (sales should be positive)"""
        clean_data = data[~pd.isna(data)]
        negative_count = np.sum(clean_data < 0)

        if negative_count > 0:
            self.warnings.append(
                f"Negative values detected: {negative_count} values. "
                f"Sales data should typically be non-negative."
            )
        self.validation_results['negative_values'] = negative_count

    def _check_outliers(self, data):
        """Check for outliers using IQR method"""
        clean_data = data[~pd.isna(data)]

        if len(clean_data) < 4:
            self.validation_results['outliers'] = 0
            return

        Q1 = np.percentile(clean_data, 25)
        Q3 = np.percentile(clean_data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = np.sum((clean_data < lower_bound) | (clean_data > upper_bound))
        outlier_pct = outliers / len(clean_data) * 100

        self.validation_results['outliers'] = outliers
        self.validation_results['outlier_percentage'] = outlier_pct

        if outliers > 0:
            self.warnings.append(
                f"Outliers detected: {outliers} values ({outlier_pct:.1f}%) outside "
                f"[{lower_bound:.2f}, {upper_bound:.2f}]. Consider investigation."
            )

    def _check_variance(self, data):
        """Check if data has sufficient variance"""
        clean_data = data[~pd.isna(data)]

        if len(clean_data) < 2:
            return

        variance = np.var(clean_data)
        std_dev = np.std(clean_data)
        mean = np.mean(clean_data)
        cv = (std_dev / mean) * 100 if mean != 0 else 0  # Coefficient of variation

        self.validation_results['variance'] = variance
        self.validation_results['std_dev'] = std_dev
        self.validation_results['coefficient_of_variation'] = cv

        if variance < 1e-10:
            self.warnings.append(
                "Very low variance detected. Data appears nearly constant, "
                "which may indicate data quality issues."
            )
        elif cv < 1:
            self.warnings.append(
                f"Low coefficient of variation: {cv:.2f}%. "
                "Limited variability may reduce model effectiveness."
            )

    def _check_stationarity(self, data):
        """Check if data is stationary using ADF test"""
        clean_data = data[~pd.isna(data)]

        if len(clean_data) < 12:
            self.warnings.append(
                "Insufficient data for stationarity test (need 12+ samples). "
                "ARIMA models assume stationarity after differencing."
            )
            return

        try:
            # Augmented Dickey-Fuller test
            adf_result = adfuller(clean_data, autolag='AIC')
            adf_statistic = adf_result[0]
            adf_pvalue = adf_result[1]

            self.validation_results['adf_statistic'] = adf_statistic
            self.validation_results['adf_pvalue'] = adf_pvalue
            self.validation_results['is_stationary'] = adf_pvalue < 0.05

            if adf_pvalue >= 0.05:
                self.warnings.append(
                    f"Data is non-stationary (ADF p-value: {adf_pvalue:.4f}). "
                    f"ARIMA will apply differencing, but consider reviewing trends."
                )

        except Exception as e:
            self.warnings.append(f"Stationarity test failed: {str(e)}")

    def _check_trend(self, data):
        """Check for trend in the data"""
        clean_data = data[~pd.isna(data)]

        if len(clean_data) < 3:
            return

        # Linear regression to detect trend
        x = np.arange(len(clean_data))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, clean_data)

        self.validation_results['trend_slope'] = slope
        self.validation_results['trend_r_squared'] = r_value ** 2
        self.validation_results['trend_p_value'] = p_value

        if p_value < 0.05 and abs(r_value) > 0.7:
            trend_direction = "increasing" if slope > 0 else "decreasing"
            self.validation_results['has_significant_trend'] = True
            self.validation_results['trend_direction'] = trend_direction
        else:
            self.validation_results['has_significant_trend'] = False

    def _check_seasonality(self, data):
        """Check for potential seasonality patterns"""
        clean_data = data[~pd.isna(data)]

        if len(clean_data) < 12:
            self.warnings.append(
                "Insufficient data for seasonality detection (need 12+ samples for quarterly patterns). "
                "SARIMA may not be optimal without clear seasonal patterns."
            )
            self.validation_results['seasonality_detected'] = False
            return

        # Simple check: compare variance within seasons vs between seasons
        # For quarterly data (periods of 4)
        if len(clean_data) >= 12:
            periods = 4
            n_seasons = len(clean_data) // periods

            seasonal_groups = [
                clean_data[i::periods][:n_seasons] for i in range(periods)
            ]

            # Calculate within-season variance
            within_var = np.mean([np.var(group) for group in seasonal_groups if len(group) > 1])
            # Calculate between-season variance (variance of seasonal means)
            seasonal_means = [np.mean(group) for group in seasonal_groups]
            between_var = np.var(seasonal_means)

            if between_var > within_var * 0.1:  # Some seasonal effect
                self.validation_results['seasonality_detected'] = True
                self.warnings.append(
                    "Potential seasonality detected. Consider using SARIMA or Holt-Winters models."
                )
            else:
                self.validation_results['seasonality_detected'] = False

    def _print_results(self):
        """Print validation results in a readable format"""
        print("\n" + "="*70)
        print("DATA VALIDATION REPORT")
        print("="*70)

        if self.errors:
            print("\nERRORS (must be fixed):")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")

        if self.warnings:
            print("\nWARNINGS (recommended to address):")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")

        if not self.errors and not self.warnings:
            print("\nAll validation checks passed!")

        print("\nDETAILED METRICS:")
        for key, value in self.validation_results.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

        print("="*70 + "\n")

    def get_recommendation(self):
        """Get recommendation based on validation results"""
        if len(self.errors) > 0:
            return "CRITICAL: Fix data errors before training models"
        elif len(self.warnings) > 3:
            return "WARNING: Multiple data quality issues detected. Model accuracy may be limited."
        elif len(self.warnings) > 0:
            return "CAUTION: Minor data quality issues detected. Review warnings."
        else:
            return "GOOD: Data quality checks passed"


def validate_training_data(data, min_samples=12, verbose=True):
    """
    Convenience function to validate training data

    Parameters:
    -----------
    data : array-like
        Time-series data to validate
    min_samples : int
        Minimum number of samples required
    verbose : bool
        Whether to print results

    Returns:
    --------
    dict : Validation results
    """
    validator = DataValidator(min_samples=min_samples)
    results = validator.validate(data, verbose=verbose)

    if verbose:
        print(f"\nRECOMMENDATION: {validator.get_recommendation()}\n")

    return results
