# AI Trends Analyzer Pro - Interactive Dashboard

## 🌐 Live Demo
**Dashboard URL**: [https://rashpinder1985.github.io/ai-trends-analyzer-pro/](https://rashpinder1985.github.io/ai-trends-analyzer-pro/)

## 📊 Features

### Interactive Visualizations
- **Historical Data Analysis** - Bar charts for AI skills distribution
- **Geographic Insights** - Doughnut charts for state-wise analysis  
- **Trend Analysis** - Line charts for salary trends over time
- **Future Predictions** - ML-based forecasting with confidence intervals

### Real-time Configuration
- Adjustable sample sizes (100-10,000 records)
- Custom date ranges (2015-2024)
- Flexible prediction horizons (1-10 years)
- Interactive parameter controls

### Production Features
- **Responsive Design** - Works on desktop, tablet, and mobile
- **Error Handling** - Graceful fallbacks and user feedback
- **Loading States** - Visual indicators for data processing
- **API Integration** - Real-time connection to backend services

## 🚀 Usage

### Demo Mode (GitHub Pages)
The hosted dashboard runs in demo mode, showcasing the UI and features. To access full functionality with real data, you need to run the API locally.

### Local Development
1. **Clone the repository**
   ```bash
   git clone https://github.com/Rashpinder1985/ai-trends-analyzer-pro.git
   cd ai-trends-analyzer-pro
   ```

2. **Start the API server**
   ```bash
   python -m uvicorn ai_trends.api.main:app --host 127.0.0.1 --port 8000
   ```

3. **Open the local dashboard**
   ```bash
   open dashboard/index.html
   ```

## 🛠️ Technical Stack
- **Frontend**: HTML5, CSS3, JavaScript ES6+
- **Charts**: Chart.js for interactive visualizations
- **Backend**: FastAPI with async/await
- **Data**: Python pandas, scikit-learn for ML
- **Deployment**: GitHub Pages, Docker support

## 📈 Data Analysis Capabilities

### Historical Analysis
- AI skills demand trends in Indian markets
- State-wise distribution of opportunities
- Salary progression over 2019-2024
- Technology adoption patterns

### Predictive Analytics  
- Future salary projections (2025-2030)
- Market growth forecasting
- Skill demand predictions
- Geographic expansion trends

### Business Intelligence
- Comprehensive reports with actionable insights
- Confidence scoring for predictions
- Market opportunity identification
- Technology investment recommendations

## 🔗 Links
- **Live Dashboard**: [https://rashpinder1985.github.io/ai-trends-analyzer-pro/](https://rashpinder1985.github.io/ai-trends-analyzer-pro/)
- **GitHub Repository**: [https://github.com/Rashpinder1985/ai-trends-analyzer-pro](https://github.com/Rashpinder1985/ai-trends-analyzer-pro)
- **API Documentation**: Available when running locally at `/docs`

---

Built with ❤️ for the Indian AI community | Hosted on GitHub Pages