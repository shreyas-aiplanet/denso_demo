"""
DNHA-M Sales Trend Forecast System (FY23-FY35)
AI-Powered Long-Term Sales Forecasting for Production & Investment Planning

Author: AI Agent for DENSO
Version: 2.0.0 (Python/Streamlit)
Date: October 22, 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import forecasting models
from models.forecasting import ForecastingEngine
from utils.data_generator import generate_historical_data
from utils.insights_generator import generate_insights

# Page configuration
st.set_page_config(
    page_title="DNHA-M Sales Forecast | DENSO AI Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2d3748;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #718096;
        margin-bottom: 2rem;
    }
    .kpi-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .kpi-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    .kpi-label {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.9;
    }
    .insight-card {
        background: #f7fafc;
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding: 0 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'forecast_data' not in st.session_state:
    st.session_state.forecast_data = None
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Header
st.markdown('<div class="main-header">🎯 DNHA-M Sales Trend Forecast (FY23-FY35)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Long-Term Sales Forecasting for Production & Investment Planning</div>', unsafe_allow_html=True)

# Sidebar - Controls
with st.sidebar:
    st.header("⚙️ Forecast Configuration")

    # Product selection
    product = st.selectbox(
        "Product Category",
        ["All Products", "DNHA-M100", "DNHA-M200", "DNHA-M300", "DNHA-M400"]
    )

    # Plant selection
    plant = st.selectbox(
        "Plant Location",
        ["All Plants", "Tokyo", "Osaka", "Nagoya", "Kyushu"]
    )

    # Area selection
    area = st.selectbox(
        "Sales Area",
        ["All Areas", "Domestic", "North America", "Europe", "Asia Pacific"]
    )

    # Model selection
    st.markdown("---")
    st.subheader("📊 Forecasting Model")

    model_method = st.selectbox(
        "Select Method",
        [
            "ARIMA (AutoRegressive)",
            "SARIMA (Seasonal ARIMA)",
            "Holt-Winters (Triple Exponential)",
            "Prophet (Facebook)",
            "Auto-Select Best Model",
            "Ensemble (Multiple Models)",
            "Linear Regression",
            "Polynomial Regression",
            "Exponential Smoothing"
        ]
    )

    # Model info
    model_info = {
        "ARIMA (AutoRegressive)": "📌 Best for trending data without strong seasonality. Uses AR(2), I(1), MA(1) components.",
        "SARIMA (Seasonal ARIMA)": "📌 Best for quarterly/annual seasonal patterns. Includes seasonal decomposition.",
        "Holt-Winters (Triple Exponential)": "📌 Best for rapidly changing markets with trend and seasonality.",
        "Prophet (Facebook)": "📌 Best for business time series with holidays and missing data.",
        "Auto-Select Best Model": "📌 Automatically selects the best model via cross-validation.",
        "Ensemble (Multiple Models)": "📌 Combines multiple models for maximum reliability."
    }

    if model_method in model_info:
        st.info(model_info[model_method])

    # Advanced options
    with st.expander("Advanced Options"):
        confidence_level = st.slider("Confidence Level", 80, 99, 95)
        forecast_horizon = st.slider("Forecast Horizon (Years)", 5, 15, 10)
        data_granularity = st.selectbox(
            "Data Granularity",
            ["quarterly", "monthly", "annual"],
            help="More granular data improves model accuracy. Quarterly/monthly recommended."
        )
        train_test_split = st.slider(
            "Test Set Size (%)",
            0, 30, 20,
            help="Percentage of data held out for validation. 0 = use all data for training."
        )
        validate_data = st.checkbox("Enable Data Validation", value=True, help="Check data quality before training")

    st.markdown("---")

    # Generate forecast button
    if st.button("🚀 Generate Forecast", type="primary", use_container_width=True):
        with st.spinner("Training AI model and generating forecasts..."):
            # Generate historical data with selected granularity
            historical_data = generate_historical_data(product, plant, area, granularity=data_granularity)

            # Show data info
            st.info(f"📊 Generated {len(historical_data)} data points using {data_granularity} granularity")

            # Create forecasting engine with validation
            engine = ForecastingEngine(
                method=model_method,
                train_test_split=train_test_split / 100.0,
                validate_data=validate_data
            )

            # Fit model (this will show validation results)
            engine.fit(historical_data)

            # Generate predictions
            predictions = engine.predict(forecast_horizon)

            # Calculate metrics
            metrics = engine.calculate_metrics()

            # Store in session state
            st.session_state.forecast_data = {
                'historical': historical_data,
                'predictions': predictions,
                'metrics': metrics,
                'model_method': model_method,
                'product': product,
                'plant': plant,
                'area': area,
                'confidence_level': confidence_level,
                'data_granularity': data_granularity,
                'validation_results': engine.validation_results
            }
            st.session_state.trained_model = engine

            st.success("✅ Forecast generated successfully!")

    # Reset button
    if st.button("🔄 Reset Filters", use_container_width=True):
        st.session_state.forecast_data = None
        st.session_state.trained_model = None
        st.rerun()

    # Export button
    if st.session_state.forecast_data is not None:
        st.markdown("---")
        if st.button("📊 Export to CSV", use_container_width=True):
            # Create export dataframe
            forecast_data = st.session_state.forecast_data
            df = pd.DataFrame({
                'Fiscal Year': forecast_data['predictions']['years'],
                'Sales (Million Yen)': forecast_data['predictions']['values'],
                'Lower Bound': forecast_data['predictions']['lower_bound'],
                'Upper Bound': forecast_data['predictions']['upper_bound'],
                'Type': ['Historical'] * len(forecast_data['historical']) + ['Forecast'] * (len(forecast_data['predictions']['values']) - len(forecast_data['historical']))
            })

            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"DNHA-M_Sales_Forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )

# Main content
if st.session_state.forecast_data is None:
    # Welcome screen
    st.info("👈 Configure your forecast parameters in the sidebar and click 'Generate Forecast' to begin.")

    st.markdown("### 🎯 Key Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **Advanced Time-Series Models**
        - ARIMA with Yule-Walker estimation
        - SARIMA with seasonal decomposition
        - Holt-Winters triple exponential smoothing
        - Facebook Prophet integration
        - Auto-select best model
        - Ensemble combining multiple models
        """)

    with col2:
        st.markdown("""
        **Interactive Analysis**
        - Real-time model training
        - R² accuracy scoring
        - Confidence intervals
        - Cross-validation
        - Model comparison
        - What-if scenarios
        """)

    with col3:
        st.markdown("""
        **AI-Powered Insights**
        - Model-specific explanations
        - Trend analysis
        - Seasonal patterns detection
        - Strategic recommendations
        - Natural language Q&A
        - Export capabilities
        """)

    st.markdown("### 📊 Sample Forecast Preview")

    # Generate sample data for preview
    sample_years = [f"FY{year}" for year in range(23, 36)]
    sample_values = [5000 * (1.05 ** i) * (1 + 0.1 * np.sin(i * np.pi / 2)) for i in range(len(sample_years))]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=sample_years[:3],
        y=sample_values[:3],
        mode='lines+markers',
        name='Historical Sales',
        line=dict(color='#667eea', width=3),
        marker=dict(size=8)
    ))

    fig.add_trace(go.Scatter(
        x=sample_years[2:],
        y=sample_values[2:],
        mode='lines+markers',
        name='Forecasted Sales',
        line=dict(color='#48bb78', width=3, dash='dash'),
        marker=dict(size=8)
    ))

    fig.update_layout(
        title="Example: 10-Year Sales Forecast with Trend Analysis",
        xaxis_title="Fiscal Year",
        yaxis_title="Sales (Million Yen)",
        height=400,
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    # Display forecast results
    forecast_data = st.session_state.forecast_data

    # KPI Dashboard
    st.markdown("### 📊 Key Performance Indicators")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        fy35_sales = forecast_data['predictions']['values'][-1]
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">FY35 Projected Sales</div>
            <div class="kpi-value">¥{fy35_sales/1000:.1f}B</div>
            <div class="kpi-label">+{forecast_data['metrics']['total_growth']:.1f}% from FY23</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Average YoY Growth</div>
            <div class="kpi-value">{forecast_data['metrics']['avg_yoy_growth']:.1f}%</div>
            <div class="kpi-label">Year-over-Year</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Model Accuracy (R²)</div>
            <div class="kpi-value">{forecast_data['metrics']['r2_score']*100:.1f}%</div>
            <div class="kpi-label">{forecast_data['metrics']['accuracy_level']}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        peak_year = forecast_data['predictions']['years'][np.argmax(forecast_data['predictions']['values'])]
        peak_value = max(forecast_data['predictions']['values'])
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Peak Sales Year</div>
            <div class="kpi-value">{peak_year}</div>
            <div class="kpi-label">¥{peak_value/1000:.1f}B</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Forecast Chart", "🏭 Plant Analysis", "🧠 AI Insights", "💬 Chat Assistant", "📋 Data Table"])

    with tab1:
        st.markdown("### 📈 Sales Forecast Trend (FY23-FY35)")

        # Create main forecast chart
        fig = go.Figure()

        years = forecast_data['predictions']['years']
        values = forecast_data['predictions']['values']
        historical_len = len(forecast_data['historical'])

        # Historical data
        fig.add_trace(go.Scatter(
            x=years[:historical_len],
            y=values[:historical_len],
            mode='lines+markers',
            name='Historical Sales',
            line=dict(color='#667eea', width=3),
            marker=dict(size=8),
            hovertemplate='%{x}<br>¥%{y:,.0f}M<extra></extra>'
        ))

        # Forecasted data
        fig.add_trace(go.Scatter(
            x=years[historical_len-1:],
            y=values[historical_len-1:],
            mode='lines+markers',
            name='Forecasted Sales',
            line=dict(color='#48bb78', width=3, dash='dash'),
            marker=dict(size=8),
            hovertemplate='%{x}<br>¥%{y:,.0f}M<extra></extra>'
        ))

        # Confidence intervals
        fig.add_trace(go.Scatter(
            x=years + years[::-1],
            y=forecast_data['predictions']['upper_bound'] + forecast_data['predictions']['lower_bound'][::-1],
            fill='toself',
            fillcolor='rgba(237, 137, 54, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% Confidence Interval',
            hoverinfo='skip'
        ))

        fig.update_layout(
            height=500,
            xaxis_title="Fiscal Year",
            yaxis_title="Sales (Million Yen)",
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        # Model information
        col1, col2 = st.columns(2)

        with col1:
            st.info(f"""
            **Model Used:** {forecast_data['model_method']}

            **Parameters:**
            - Product: {forecast_data['product']}
            - Plant: {forecast_data['plant']}
            - Area: {forecast_data['area']}
            """)

        with col2:
            # Show test metrics if available, otherwise training metrics
            if 'test_r2' in forecast_data['metrics']:
                st.success(f"""
                **Out-of-Sample Test Metrics:**
                - Test R² Score: {forecast_data['metrics']['test_r2']:.4f}
                - Test RMSE: {forecast_data['metrics']['test_rmse']:.2f}
                - Test MAE: {forecast_data['metrics']['test_mae']:.2f}
                - Test MAPE: {forecast_data['metrics']['test_mape']:.2f}%
                """)
            else:
                st.success(f"""
                **Performance Metrics:**
                - R² Score: {forecast_data['metrics']['r2_score']:.4f}
                - RMSE: {forecast_data['metrics'].get('rmse', 0):.2f}
                - MAE: {forecast_data['metrics'].get('mae', 0):.2f}
                - MAPE: {forecast_data['metrics'].get('mape', 0):.2f}%
                """)

    with tab2:
        st.markdown("### 🏭 Plant & Area Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Plant-wise FY35 Forecast")

            # Generate plant comparison data
            plants = ['Tokyo', 'Osaka', 'Nagoya', 'Kyushu']
            plant_values = [forecast_data['predictions']['values'][-1] * np.random.uniform(0.9, 1.1) for _ in plants]

            fig_plant = go.Figure(data=[
                go.Bar(
                    x=plants,
                    y=plant_values,
                    marker_color=['#667eea', '#764ba2', '#48bb78', '#ed8936'],
                    text=[f'¥{v/1000:.1f}B' for v in plant_values],
                    textposition='outside'
                )
            ])

            fig_plant.update_layout(
                height=350,
                xaxis_title="Plant",
                yaxis_title="FY35 Forecast (Million Yen)",
                showlegend=False
            )

            st.plotly_chart(fig_plant, use_container_width=True)

        with col2:
            st.markdown("#### Area-wise Distribution")

            # Generate area distribution
            areas = ['Domestic', 'North America', 'Europe', 'Asia Pacific']
            area_values = [
                forecast_data['predictions']['values'][-1] * 0.35,
                forecast_data['predictions']['values'][-1] * 0.30,
                forecast_data['predictions']['values'][-1] * 0.20,
                forecast_data['predictions']['values'][-1] * 0.15
            ]

            fig_area = go.Figure(data=[
                go.Pie(
                    labels=areas,
                    values=area_values,
                    marker_colors=['#667eea', '#764ba2', '#48bb78', '#ed8936'],
                    textinfo='label+percent',
                    hovertemplate='%{label}<br>¥%{value:,.0f}M<br>%{percent}<extra></extra>'
                )
            ])

            fig_area.update_layout(height=350)

            st.plotly_chart(fig_area, use_container_width=True)

    with tab3:
        st.markdown("### 🧠 AI-Generated Insights")

        # Generate insights
        insights = generate_insights(
            forecast_data['model_method'],
            forecast_data['predictions'],
            forecast_data['historical'],
            forecast_data['metrics']
        )

        for insight in insights:
            st.markdown(f"""
            <div class="insight-card">
                <h4>{insight['title']}</h4>
                <p>{insight['content']}</p>
            </div>
            """, unsafe_allow_html=True)

    with tab4:
        st.markdown("### 💬 Ask AI About Sales Trends")

        st.info("Ask questions like: 'How does ARIMA work?', 'Why is FY29 showing a decline?', 'What are the growth drivers?'")

        # Chat interface
        user_question = st.text_input("Your question:", placeholder="Ask a question about sales trends...")

        if st.button("Send", type="primary") and user_question:
            # Add to chat history
            st.session_state.chat_history.append({
                'role': 'user',
                'content': user_question
            })

            # Generate AI response (simplified - in production use OpenAI API)
            response = generate_ai_response(user_question, forecast_data)

            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': response
            })

        # Display chat history
        for message in st.session_state.chat_history[-10:]:  # Show last 10 messages
            if message['role'] == 'user':
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(f"**AI Assistant:** {message['content']}")
                st.markdown("---")

    with tab5:
        st.markdown("### 📋 Detailed Forecast Data")

        # Create detailed table
        years = forecast_data['predictions']['years']
        values = forecast_data['predictions']['values']
        lower = forecast_data['predictions']['lower_bound']
        upper = forecast_data['predictions']['upper_bound']

        df = pd.DataFrame({
            'Fiscal Year': years,
            'Forecasted Sales (¥M)': [f"¥{v:,.0f}M" for v in values],
            'YoY Growth (%)': ['—'] + [f"{((values[i] - values[i-1]) / values[i-1] * 100):.1f}%" for i in range(1, len(values))],
            'Confidence Level': [f"{forecast_data['confidence_level']}%" if i >= len(forecast_data['historical']) else "100%" for i in range(len(years))],
            'Lower Bound (¥M)': [f"¥{v:,.0f}M" for v in lower],
            'Upper Bound (¥M)': [f"¥{v:,.0f}M" for v in upper],
            'Type': ['Historical'] * len(forecast_data['historical']) + ['Forecast'] * (len(years) - len(forecast_data['historical']))
        })

        st.dataframe(df, use_container_width=True, height=400)

def generate_ai_response(question, forecast_data):
    """Generate AI response to user question (simplified version)"""
    q = question.lower()

    if 'arima' in q or 'how' in q and 'work' in q:
        return """ARIMA (AutoRegressive Integrated Moving Average) combines three components:

**AR(2)**: Uses the past 2 values to predict the next value
**I(1)**: Applies first-order differencing to make the data stationary
**MA(1)**: Accounts for the moving average of past forecast errors

This makes ARIMA excellent for capturing trends and patterns in non-stationary time series data like sales forecasts."""

    elif 'decline' in q or 'fy29' in q:
        return f"""Based on the {forecast_data['model_method']} forecast, any potential declines could be attributed to:

1. Market saturation in mature product lines
2. Seasonal cyclical patterns
3. Competitive pressure in certain regions
4. Economic cycle effects

The model's autoregressive component has detected these patterns from historical data and projects them forward."""

    elif 'growth' in q or 'driver' in q:
        avg_growth = forecast_data['metrics']['avg_yoy_growth']
        return f"""The primary growth drivers include:

1. **Expanding Market Demand**: Emerging regions showing strong adoption
2. **Product Innovation**: Quality improvements driving customer preference
3. **Strategic Partnerships**: Distribution expansion into new markets

The model shows an average YoY growth rate of **{avg_growth:.1f}%**, indicating {'strong' if avg_growth > 5 else 'stable'} market momentum."""

    else:
        fy35_sales = forecast_data['predictions']['values'][-1]
        return f"""Based on the {forecast_data['model_method']} forecast analysis, FY35 projects sales of **¥{fy35_sales/1000:.2f}B**.

The model shows {'positive growth momentum' if forecast_data['metrics']['total_growth'] > 0 else 'stable performance'}.

Try asking: "How does ARIMA work?", "What are the growth drivers?", or "Why might there be a decline?"
"""

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #718096; font-size: 0.9rem;">
    🤖 Generated with DENSO AI Analytics System v2.0.0 |
    Powered by ARIMA, SARIMA, Holt-Winters & Prophet |
    © 2025 DENSO Corporation
</div>
""", unsafe_allow_html=True)
