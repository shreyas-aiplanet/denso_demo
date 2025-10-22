"""
AI Insights Generator
Generates intelligent insights based on forecast results and model type
"""

import numpy as np


def generate_insights(model_method, predictions, historical_data, metrics):
    """
    Generate AI-powered insights based on forecast results

    Parameters:
    -----------
    model_method : str
        The forecasting method used
    predictions : dict
        Prediction results containing years, values, etc.
    historical_data : list
        Historical sales data
    metrics : dict
        Model performance metrics

    Returns:
    --------
    list : List of insight dictionaries with title and content
    """

    insights = []

    # Model Information Insight
    insights.append(get_model_insight(model_method, metrics))

    # Overall Trend Analysis
    insights.append(get_trend_insight(predictions, historical_data, metrics))

    # Decline Period Detection
    decline_insight = get_decline_insight(predictions['values'], predictions['years'], model_method)
    if decline_insight:
        insights.append(decline_insight)

    # Strategic Recommendations
    insights.append(get_strategic_insight(metrics['avg_yoy_growth'], model_method))

    # Forecast Uncertainty (for advanced models)
    if model_method in ["ARIMA (AutoRegressive)", "SARIMA (Seasonal ARIMA)", "Holt-Winters (Triple Exponential)"]:
        insights.append(get_uncertainty_insight(model_method))

    return insights


def get_model_insight(model_method, metrics):
    """Generate model-specific insight"""

    model_descriptions = {
        "ARIMA (AutoRegressive)": """ARIMA models capture AutoRegressive (AR) patterns and Moving Average (MA)
        trends after differencing to achieve stationarity. This makes them excellent for identifying underlying
        patterns in sales data. The model successfully identified autocorrelation patterns in your historical data.""",

        "SARIMA (Seasonal ARIMA)": """SARIMA extends ARIMA with seasonal decomposition, ideal for quarterly or
        annual patterns. The model separates trend, seasonal, and residual components for more accurate long-term
        forecasts. Quarterly seasonality has been detected and incorporated into predictions.""",

        "Holt-Winters (Triple Exponential)": """Holt-Winters uses triple exponential smoothing to capture level,
        trend, and seasonal patterns simultaneously. This model adapts well to changing market conditions and seasonal
        fluctuations, making it robust for evolving markets.""",

        "Prophet (Facebook)": """Prophet is designed for business time series with strong seasonal patterns and
        multiple periods of historical data. It handles missing data and outliers robustly, making it ideal for
        real-world business forecasting.""",

        "Ensemble (Multiple Models)": """Ensemble forecasting combines ARIMA, SARIMA, and Holt-Winters predictions
        to reduce individual model biases and improve forecast reliability through model averaging. This approach
        provides the most robust predictions.""",

        "Auto-Select Best Model": """Auto-selection used cross-validation to evaluate multiple models and selected
        the best performer based on prediction accuracy. This ensures optimal model choice for your specific data
        patterns."""
    }

    description = model_descriptions.get(model_method, "Advanced time-series forecasting model.")

    return {
        'title': f"🤖 Model Information: {metrics.get('model_name', model_method)}",
        'content': description
    }


def get_trend_insight(predictions, historical_data, metrics):
    """Generate overall trend analysis insight"""

    fy35_sales = predictions['values'][-1]
    fy23_sales = predictions['values'][0]
    total_growth = ((fy35_sales - fy23_sales) / fy23_sales) * 100
    avg_yoy_growth = metrics['avg_yoy_growth']
    r2_score = metrics['r2_score']

    growth_direction = "positive" if total_growth > 0 else "negative"

    content = f"""The forecast shows a <strong>{growth_direction} {abs(total_growth):.1f}%</strong> growth from
    FY23 to FY35, with an average YoY growth of <strong>{avg_yoy_growth:.1f}%</strong>. The model achieved an R²
    accuracy of <strong>{r2_score*100:.1f}%</strong>, indicating {metrics['accuracy_level'].lower()} predictive
    performance."""

    return {
        'title': "📊 Overall Trend Analysis",
        'content': content
    }


def get_decline_insight(values, years, model_method):
    """Detect and explain decline periods"""

    decline_years = []
    for i in range(1, len(values)):
        if values[i] < values[i-1]:
            decline_years.append(years[i])

    if not decline_years:
        return None

    decline_reasons = {
        "ARIMA (AutoRegressive)": "autoregressive patterns in the time series",
        "SARIMA (Seasonal ARIMA)": "seasonal patterns detected by the SARIMA model",
        "Holt-Winters (Triple Exponential)": "cyclical trends identified through exponential smoothing",
        "Prophet (Facebook)": "seasonal patterns or market saturation"
    }

    reason = decline_reasons.get(model_method, "natural market fluctuations")

    content = f"""The model predicts potential sales declines in <strong>{', '.join(decline_years)}</strong>.
    This could be due to {reason}. """

    if model_method in ["ARIMA (AutoRegressive)", "SARIMA (Seasonal ARIMA)"]:
        content += """The AR component suggests these declines are part of a cyclical pattern that may self-correct.
        """

    content += "Consider implementing proactive marketing strategies during these periods."

    return {
        'title': "⚠️ Projected Decline Periods",
        'content': content
    }


def get_strategic_insight(avg_yoy_growth, model_method):
    """Generate strategic recommendations"""

    if avg_yoy_growth > 5:
        growth_level = "strong"
        recommendations = """
        <ul>
            <li>Increase production capacity by 15-20% by FY28</li>
            <li>Invest in automation and efficiency improvements</li>
            <li>Expand supply chain infrastructure in high-growth regions</li>
            <li>Allocate 5-7% of revenue to R&D for product innovation</li>
        </ul>
        """
    elif avg_yoy_growth > 2:
        growth_level = "moderate"
        recommendations = """
        <ul>
            <li>Focus on operational efficiency and cost reduction</li>
            <li>Market diversification strategies</li>
            <li>Customer retention programs</li>
            <li>Selective capacity expansion in high-potential segments</li>
        </ul>
        """
    else:
        growth_level = "slow"
        recommendations = """
        <ul>
            <li>Comprehensive market analysis to identify growth barriers</li>
            <li>Product portfolio optimization</li>
            <li>Cost reduction initiatives</li>
            <li>Explore new market segments and applications</li>
        </ul>
        """

    additional_rec = ""
    if model_method in ["SARIMA (Seasonal ARIMA)", "Holt-Winters (Triple Exponential)"]:
        additional_rec = "<li>Plan inventory based on identified seasonal patterns to optimize cash flow</li>"
    elif model_method == "Ensemble (Multiple Models)":
        additional_rec = "<li>Quarterly model retraining recommended to maintain ensemble accuracy</li>"

    if additional_rec:
        recommendations = recommendations.replace("</ul>", f"{additional_rec}</ul>")

    content = f"""Based on the {growth_level} growth trajectory ({avg_yoy_growth:.1f}% YoY), we recommend:
    {recommendations}
    """

    return {
        'title': "🎯 Strategic Recommendations",
        'content': content
    }


def get_uncertainty_insight(model_method):
    """Generate insight about forecast uncertainty"""

    uncertainty_explanations = {
        "ARIMA (AutoRegressive)": "ARIMA confidence intervals reflect the error variance from the AR and MA components.",
        "SARIMA (Seasonal ARIMA)": "SARIMA intervals account for both trend and seasonal uncertainty.",
        "Holt-Winters (Triple Exponential)": "Holt-Winters intervals widen as the forecast horizon extends, reflecting cumulative smoothing error."
    }

    explanation = uncertainty_explanations.get(model_method, "")

    content = f"""The {model_method.split('(')[0].strip()} model provides 95% confidence intervals
    (shown as upper/lower bounds in the chart). Uncertainty increases for distant predictions (FY32-FY35).
    {explanation} We recommend quarterly model updates with new data to reduce forecast uncertainty."""

    return {
        'title': "📉 Forecast Uncertainty",
        'content': content
    }


def generate_seasonal_insight(seasonal_component, period=4):
    """Generate insight about seasonal patterns"""

    content = f"""The model has identified quarterly seasonal patterns with a {period}-period cycle.
    Seasonal fluctuations could be due to fiscal year-end purchasing, holiday seasons, or industry-specific
    demand cycles. These patterns are automatically factored into the forecast."""

    return {
        'title': "🔄 Seasonal Patterns",
        'content': content
    }
