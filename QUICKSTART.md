# Quick Start Guide - DNHA-M Sales Forecast (Streamlit)

## 🚀 Get Started in 3 Minutes

### Step 1: Install Dependencies (1 minute)

Open terminal/command prompt in the project directory:

```bash
cd D:\dev-aiplanet\denso_demo
pip install -r requirements.txt
```

### Step 2: Run the Application (10 seconds)

```bash
streamlit run app.py
```

Your browser will automatically open to `http://localhost:8501`

### Step 3: Generate Your First Forecast (30 seconds)

1. **In the sidebar**, leave filters as default:
   - Product: "All Products"
   - Plant: "All Plants"
   - Area: "All Areas"
   - Model: "ARIMA (AutoRegressive)"

2. **Click** the "🚀 Generate Forecast" button

3. **Wait** 2-3 seconds for model training

4. **View** your forecast results!

## 📊 What You'll See

### KPI Dashboard
- **FY35 Projected Sales**: Total sales forecast for 2035
- **Average YoY Growth**: Annual growth rate percentage
- **Model Accuracy (R²)**: How well the model fits (>90% = Excellent)
- **Peak Sales Year**: Year with highest sales projection

### Interactive Chart
- Blue line: Historical sales (FY23-FY25)
- Green dashed line: Forecasted sales (FY26-FY35)
- Orange shaded area: 95% confidence interval

### 5 Tabs to Explore

1. **📈 Forecast Chart**: Main sales trend visualization
2. **🏭 Plant Analysis**: Compare plants and regions
3. **🧠 AI Insights**: Read model explanations and recommendations
4. **💬 Chat Assistant**: Ask questions about the forecast
5. **📋 Data Table**: Detailed year-by-year breakdown

## 🎯 Try Different Models

### ARIMA (Default)
Best for general trends. Fast and accurate.

### SARIMA
Best if you see quarterly patterns. Try it and compare!

```
Sidebar → Select "SARIMA (Seasonal ARIMA)" → Generate Forecast
```

### Ensemble
Want maximum reliability? Combines 4 models.

```
Sidebar → Select "Ensemble (Multiple Models)" → Generate Forecast
```

### Auto-Select
Let AI choose the best model for you.

```
Sidebar → Select "Auto-Select Best Model" → Generate Forecast
```

## 💬 Ask the AI Questions

Go to **💬 Chat Assistant** tab and ask:

- "How does ARIMA work?"
- "Why is there a decline in FY29?"
- "What are the main growth drivers?"
- "Which model should I use?"

## 📊 Export Your Forecast

1. Generate a forecast
2. Sidebar → Click "📊 Export to CSV"
3. Click "Download CSV"
4. Open in Excel or Google Sheets

## 🎛️ Advanced: Try Different Filters

### Compare Products

```
Product: "DNHA-M100" → Generate Forecast
Product: "DNHA-M300" → Generate Forecast
```

See how different product lines perform!

### Compare Plants

```
Plant: "Tokyo" → Generate Forecast
Plant: "Nagoya" → Generate Forecast
```

Which plant has better growth potential?

### Compare Regions

```
Area: "Domestic" → Generate Forecast
Area: "North America" → Generate Forecast
```

Where should you invest?

## 🔍 Understanding Results

### R² Score (Model Accuracy)
- **>90%**: Excellent - Trust the forecast
- **70-90%**: Good - Reliable predictions
- **<70%**: Fair - Use with caution

### YoY Growth
- **>5%**: Strong growth - Plan for expansion
- **2-5%**: Moderate growth - Steady investments
- **<2%**: Slow growth - Focus on efficiency

### Confidence Intervals
- Narrow bands: High confidence
- Wide bands: More uncertainty
- Gets wider for distant years (FY33-35)

## ⚠️ Troubleshooting

### Port Already in Use?

```bash
streamlit run app.py --server.port 8502
```

### Prophet Installation Issues on Windows?

```bash
# Install Visual C++ Build Tools first, then:
conda install -c conda-forge prophet
```

Or skip Prophet and use ARIMA/SARIMA instead!

### Import Errors?

```bash
pip install --upgrade -r requirements.txt
```

## 🎓 Next Steps

1. ✅ Generated your first forecast
2. 📖 Read `README_STREAMLIT.md` for full documentation
3. 🔬 Read `ARIMA_MODELS_GUIDE.md` to understand the models
4. 🎯 Try all 9 forecasting models and compare results
5. 💼 Present insights to stakeholders

## 📞 Need Help?

- Check `README_STREAMLIT.md` for detailed documentation
- Review `ARIMA_MODELS_GUIDE.md` for model explanations
- Create a GitHub issue for bugs

## 🎉 Success!

You're now ready to generate professional sales forecasts with advanced AI models!

---

**Time to first forecast**: < 3 minutes
**Difficulty**: Beginner-friendly
**Power**: Production-grade time-series forecasting
