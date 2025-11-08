import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time

# Load trained model
# Load trained model
import joblib

model = joblib.load("best_rf_model.joblib")

# No encoders.pkl used — provide fallback frequency maps
product_freq_map = {}
gpu_model_freq_map = {}

# Page Configuration
st.set_page_config(
    page_title="Laptop Price Predictor ",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling with new color scheme
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        padding: 2rem;
    }
    
    /* Card-like container for content */
    .stApp {
        background-color: #ffffff;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        margin: -1rem -1rem 2rem -1rem;
        box-shadow: 0 10px 40px rgba(0, 210, 255, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .header-container::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: pulse 4s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 0.8; }
    }
    
    .main-title {
        color: white;
        font-size: 3.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-align: center;
        text-shadow: 3px 3px 6px rgba(0, 0, 0, 0.3);
        position: relative;
        z-index: 1;
    }
    
    .subtitle {
        color: rgba(255, 255, 255, 0.95);
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 0;
        position: relative;
        z-index: 1;
        font-weight: 400;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f7f9fc 0%, #e8f1f8 100%);
    }
    
    [data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(180deg, #f7f9fc 0%, #e8f1f8 100%);
    }
    
    /* Section headers in sidebar */
    .section-header {
        background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 100%);
        color: white;
        padding: 0.85rem 1.2rem;
        border-radius: 12px;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 4px 15px rgba(0, 210, 255, 0.25);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 100%);
        color: white;
        border: none;
        padding: 1rem 2.5rem;
        font-size: 1.2rem;
        font-weight: 700;
        border-radius: 15px;
        width: 100%;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: 0 6px 25px rgba(0, 210, 255, 0.4);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 10px 35px rgba(0, 210, 255, 0.6);
        background: linear-gradient(135deg, #3a7bd5 0%, #00d2ff 100%);
    }
    
    /* Result card styling */
    .result-card {
        background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 100%);
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 15px 40px rgba(0, 210, 255, 0.4);
        margin: 2rem 0;
        position: relative;
        overflow: hidden;
    }
    
    .result-card {
    background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 100%);
    padding: 2.5rem;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0 15px 40px rgba(0, 210, 255, 0.4);
    margin: 2rem 0;
    position: relative;
    transform: scale(0.95);
    opacity: 0;
    animation: fadeInScale 0.8s ease-out forwards;
        }

    @keyframes fadeInScale {
    0% { opacity: 0; transform: scale(0.8); }
    100% { opacity: 1; transform: scale(1); }
}

    .result-card:hover {
    transform: scale(1.03);
    transition: all 0.3s ease-in-out;
    }
    
    .price-label {
        color: rgba(255, 255, 255, 0.95);
        font-size: 1.3rem;
        margin-bottom: 0.8rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 2px;
        position: relative;
        z-index: 1;
    }
    
    .price-value {
        color: white;
        font-size: 3.8rem;
        font-weight: 800;
        text-shadow: 3px 3px 6px rgba(0, 0, 0, 0.3);
        position: relative;
        z-index: 1;
        animation: priceReveal 0.6s ease-out;
    }
    
    @keyframes priceReveal {
        0% { transform: scale(0.5); opacity: 0; }
        100% { transform: scale(1); opacity: 1; }
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #f8fbff 0%, #e8f4f8 100%);
        border-left: 5px solid #00d2ff;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    
    .info-box:hover {
        transform: translateX(5px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.08);
    }
    
    /* Stats cards */
    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        text-align: center;
        border: 2px solid #e8f4f8;
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(0, 210, 255, 0.2);
        border-color: #00d2ff;
    }
    
    .stat-icon {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .stat-label {
        color: #6c757d;
        font-size: 0.9rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stat-value {
        color: #2c5364;
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 0.3rem;
    }
    
    /* Comparison section */
    .comparison-box {
        background: linear-gradient(135deg, #fff5e6 0%, #ffe8cc 100%);
        border-left: 5px solid #ff9500;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(255, 149, 0, 0.1);
    }
    
    /* Price range indicator */
    .price-indicator {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin: 1.5rem 0;
        padding: 1rem;
        background: white;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    }
    
    .price-point {
        text-align: center;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 600;
    }
    
    .price-point.budget {
        background: #e8f5e9;
        color: #2e7d32;
    }
    
    .price-point.mid {
        background: #fff3e0;
        color: #e65100;
    }
    
    .price-point.premium {
        background: #f3e5f5;
        color: #6a1b9a;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Adjust sidebar width */
    [data-testid="stSidebar"][aria-expanded="true"] {
        min-width: 380px;
        max-width: 380px;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 100%);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #3a7bd5 0%, #00d2ff 100%);
    }
            
    /* === Text Visibility Fix === */

    /* Make text inside gradient boxes darker and readable */
    .result-card .price-value,
    .result-card .price-label,
    .comparison-box h3,
    .comparison-box p,
    .section-header,
    .info-box,
    .price-point h4,
    .price-point p,
    div, p, span {
        color: #1a1a1a !important;  /* deep charcoal text */
    }

    /* Give subtle text shadow on gradient backgrounds */
    .result-card .price-value,
    .comparison-box h3 {
        text-shadow: 1px 1px 3px rgba(0,0,0,0.25);
    }

    /* Increase brightness/contrast for all content */
    .result-card, .comparison-box {
        filter: brightness(1.1) contrast(1.15);
    }

    /* Chart title + labels darker */
    .js-plotly-plot .plotly text {
        fill: #1a1a1a !important;
    }

    /* Fix faded chart title */
    .gtitle {
        fill: #1a1a1a !important;
        font-weight: 600 !important;
    }

    /* Headings across app */
    h1, h2, h3, h4, h5, h6 {
        color: #1a1a1a !important;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="header-container">
        <h1 class="main-title">💻 Laptop Price Predictor</h1>
        <p class="subtitle">Instant price estimates powered by advanced machine learning algorithms</p>
    </div>
""", unsafe_allow_html=True)

# Load dataset and populate dropdowns dynamically

try:
    df = pd.read_csv("laptop_prices.csv")
except FileNotFoundError:
    st.error("⚠️ 'laptop_prices.csv' not found. Please place it in the same directory as app.py")
    st.stop()

df.columns = df.columns.str.strip()

# Extract unique dropdown values
company_options = sorted(df['Company'].dropna().unique())
typename_options = sorted(df['TypeName'].dropna().unique())
cpu_company_options = sorted(df['CPU_company'].dropna().unique())

# Create mapping: company → available products
brand_models = {
    brand: sorted(df[df['Company'] == brand]['Product'].dropna().unique().tolist())
    for brand in company_options
}

# Create mapping: company → available OS options
os_by_brand = {
    brand: sorted(df[df['Company'] == brand]['OS'].dropna().unique().tolist())
    for brand in company_options
}

# Create mapping: company → available TypeName options
type_by_brand = {
    brand: sorted(df[df['Company'] == brand]['TypeName'].dropna().unique().tolist())
    for brand in company_options
}

# Create mapping: GPU company → available GPU models
gpu_by_company = {
    company: sorted(df[df['GPU_company'] == company]['GPU_model'].dropna().unique().tolist())
    for company in ['Intel', 'Nvidia', 'AMD']
}
# Sidebar with organized sections
st.sidebar.image("https://img.icons8.com/clouds/200/laptop.png", use_column_width=True)
st.sidebar.markdown("<h2 style='text-align: center; color: #00d2ff; margin-bottom: 2rem; font-weight: 700;'>Configure Your Laptop</h2>", unsafe_allow_html=True)

# Brand first
st.sidebar.markdown("<div class='section-header'><i class='fas fa-building'></i> Select Brand</div>", unsafe_allow_html=True)
brand = st.sidebar.selectbox("Brand", sorted(df['Company'].dropna().unique()))

# Then filter models for that brand
st.sidebar.markdown("<div class='section-header'><i class='fas fa-laptop'></i> Select Model</div>", unsafe_allow_html=True)
selected_model = st.sidebar.selectbox("Model", sorted(df[df['Company'] == brand]['Product'].dropna().unique()))

# Filter row corresponding to selected model
model_row = df[(df['Company'] == brand) & (df['Product'] == selected_model)].iloc[0] if not df[(df['Company'] == brand) & (df['Product'] == selected_model)].empty else None

# Auto-fill related fields from dataset
typename = st.sidebar.selectbox("Type", [model_row['TypeName']] if model_row is not None else sorted(df['TypeName'].dropna().unique()))
os = st.sidebar.selectbox("Operating System", [model_row['OS']] if model_row is not None else sorted(df['OS'].dropna().unique()))
cpu_company = st.sidebar.selectbox("CPU Company", [model_row['CPU_company']] if model_row is not None else sorted(df['CPU_company'].dropna().unique()))
gpu_company = st.sidebar.selectbox("GPU Company", [model_row['GPU_company']] if model_row is not None else sorted(df['GPU_company'].dropna().unique()))
gpu_model = st.sidebar.selectbox("GPU Model", [model_row['GPU_model']] if model_row is not None else sorted(df['GPU_model'].dropna().unique()))
storage_type = st.sidebar.selectbox("Primary Storage Type", [model_row['PrimaryStorageType']] if model_row is not None else sorted(df['PrimaryStorageType'].dropna().unique()))
secondary_storage = st.sidebar.selectbox("Secondary Storage", [model_row['SecondaryStorageType']] if model_row is not None else sorted(df['SecondaryStorageType'].dropna().unique()))

# Fill numeric sliders with dataset defaults
ram = st.sidebar.slider("RAM (GB)", 2, 64, int(model_row['Ram']) if model_row is not None else 8, step=2)
cpu_freq = st.sidebar.slider("CPU Frequency (GHz)", 1.0, 5.0, float(model_row['CPU_freq']) if model_row is not None else 2.5, 0.1)
total_storage = st.sidebar.slider("Total Storage (GB)", 128, 4096, int(model_row['PrimaryStorage'] + model_row['SecondaryStorage']) if model_row is not None else 512, step=128)
ppi = st.sidebar.slider("Screen PPI", 90, 350, 140, step=5)
weight = st.sidebar.slider("Weight (kg)", 0.8, 4.0, float(model_row['Weight']) if model_row is not None else 2.0, 0.1)

# Display features
touchscreen = st.sidebar.selectbox("Touchscreen", [model_row['Touchscreen']] if model_row is not None else ['Yes', 'No'])
ips_panel = st.sidebar.selectbox("IPS Panel", [model_row['IPSpanel']] if model_row is not None else ['Yes', 'No'])
retina = st.sidebar.selectbox("Retina Display", [model_row['RetinaDisplay']] if model_row is not None else ['Yes', 'No'])

# Main content area with tabs
tab1, tab2, tab3 = st.tabs(["📊 Configuration Summary", "💰 Price Prediction", "📈 Market Analysis"])

with tab1:
    st.markdown("### Your Laptop Configuration")
    
    # Display specs in cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
            <div class='stat-card'>
                <div class='stat-icon'>🏢</div>
                <div class='stat-label'>Brand</div>
                <div class='stat-value'>{brand}</div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class='stat-card'>
                <div class='stat-icon'>💻</div>
                <div class='stat-label'>Type</div>
                <div class='stat-value'>{typename}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class='stat-card'>
                <div class='stat-icon'>⚡</div>
                <div class='stat-label'>CPU</div>
                <div class='stat-value'>{cpu_company}</div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class='stat-card'>
                <div class='stat-icon'>🎮</div>
                <div class='stat-label'>GPU</div>
                <div class='stat-value'>{gpu_company}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class='stat-card'>
                <div class='stat-icon'>🧠</div>
                <div class='stat-label'>RAM</div>
                <div class='stat-value'>{ram} GB</div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class='stat-card'>
                <div class='stat-icon'>💾</div>
                <div class='stat-label'>Storage</div>
                <div class='stat-value'>{total_storage} GB</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Detailed specs
    st.markdown("<div class='info-box'>", unsafe_allow_html=True)
    st.markdown("**📋 Detailed Specifications**")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Model:** {selected_model}")
        st.markdown(f"**Operating System:** {os}")
        st.markdown(f"**CPU Frequency:** {cpu_freq} GHz")
        st.markdown(f"**Primary Storage:** {storage_type}")
        st.markdown(f"**Display PPI:** {ppi}")
    with col2:
        st.markdown(f"**GPU Model:** {gpu_model}")
        st.markdown(f"**Secondary Storage:** {secondary_storage}")
        st.markdown(f"**Weight:** {weight} kg")
        st.markdown(f"**Touchscreen:** {touchscreen}")
        st.markdown(f"**IPS Panel:** {ips_panel} | **Retina:** {retina}")
    st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown("### Get Your Price Estimate")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Prepare input data
        input_data = pd.DataFrame([{
            'Company': brand,
            'Product': selected_model,
            'TypeName': typename,
            'OS': os,
            'PrimaryStorageType': storage_type,
            'SecondaryStorageType': secondary_storage,
            'GPU_company': gpu_company,
            'GPU_model': gpu_model,
            'CPU_company': cpu_company,
            'Touchscreen': touchscreen,
            'IPSpanel': ips_panel,
            'RetinaDisplay': retina,
            'Ram': ram,
            'CPU_freq': cpu_freq,
            'PPI': ppi,
            'TotalStorage': total_storage,
            'Weight': weight
        }])
        
        # Fill frequency-encoded features from maps when available (applies to every input)
        input_data['Product_freq'] = product_freq_map.get(selected_model, 0.01)
        input_data['GPU_model_freq'] = gpu_model_freq_map.get(gpu_model, 0.01)
        
        # Predict button
        if st.button("🔮 Calculate Price Estimate"):
            with st.spinner("🤖 analyzing your configuration..."):
                time.sleep(1.5)  # Dramatic effect
                predicted_log = model.predict(input_data)[0]
                predicted_price = np.expm1(predicted_log)
                
                st.markdown(f"""
                    <div class='result-card'>
                        <div class='price-label'>Estimated Laptop Price</div>
                        <div class='price-value'>€{predicted_price:,.2f}</div>
                    </div>
                """, unsafe_allow_html=True)
                
                
                
                # Price category
                if predicted_price < 500:
                    category = "Budget-Friendly"
                    icon = "💚"
                    description = "Perfect for students and basic computing needs"
                elif predicted_price < 1000:
                    category = "Mid-Range"
                    icon = "💙"
                    description = "Great balance of performance and value"
                elif predicted_price < 1500:
                    category = "High-Performance"
                    icon = "💜"
                    description = "Professional-grade for demanding tasks"
                else:
                    category = "Premium"
                    icon = "💎"
                    description = "Top-tier specifications for power users"
                
                st.markdown(f"""
                    <div class='comparison-box'>
                        <h3>{icon} {category} Laptop</h3>
                        <p style='margin: 0; font-size: 1.1rem;'>{description}</p>
                    </div>
                """, unsafe_allow_html=True)

                # ======== PRICE COMPARISON CHART =========
                import plotly.graph_objects as go

                if 'Price_euros' in df.columns:
                    brand_type_data = df[(df['Company'] == brand) & (df['TypeName'] == typename)]
                    if not brand_type_data.empty:
                        avg_price = brand_type_data['Price_euros'].mean()
                        min_price = brand_type_data['Price_euros'].min()
                        max_price = brand_type_data['Price_euros'].max()

                        labels = ['Predicted Price', f'{brand} Avg ({typename})', 'Min Market', 'Max Market']
                        values = [predicted_price, avg_price, min_price, max_price]
                        colors = ['#00d2ff', '#3a7bd5', '#6c757d', '#adb5bd']

                        fig = go.Figure(
                            data=[
                                go.Bar(
                                    x=values,
                                    y=labels,
                                    orientation='h',
                                    marker_color=colors,
                                    text=[f"€{v:,.0f}" for v in values],
                                    textposition="outside"
                                )
                            ]
                        )

                        fig.update_layout(
                            title=f"💹 Price Comparison for {brand} {typename}",
                            xaxis_title="Price (€)",
                            yaxis_title="",
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            font=dict(family="Inter, sans-serif", size=13, color="#2c5364"),
                            margin=dict(l=100, r=50, t=70, b=50),
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        
                
                # Additional insights
                st.markdown("<div class='info-box'>", unsafe_allow_html=True)
                st.markdown("**💡 Smart Recommendations**")
                
                recommendations = []
                if ram < 8:
                    recommendations.append("• Consider upgrading to 8GB+ RAM for better multitasking")
                if storage_type == "HDD":
                    recommendations.append("• Switching to SSD would significantly improve performance")
                if cpu_freq < 2.0:
                    recommendations.append("• A faster CPU would enhance overall system responsiveness")
                if ppi < 120:
                    recommendations.append("• Higher PPI display would provide sharper visuals")
                
                if recommendations:
                    for rec in recommendations:
                        st.markdown(rec)
                else:
                    st.markdown("✅ Your configuration is well-balanced!")
                
                st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    st.markdown("### Market Price Analysis")
    
    st.markdown("""
        <div class='info-box'>
            <h3>📊 Price Range Guide</h3>
            <p>Understanding where your laptop fits in the market</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class='price-point budget'>
                <h4>Budget</h4>
                <p style='font-size: 1.5rem; margin: 0.5rem 0;'>€300-€700</p>
                <p style='font-size: 0.85rem; margin: 0;'>Basic tasks & browsing</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class='price-point mid'>
                <h4>Mid-Range</h4>
                <p style='font-size: 1.5rem; margin: 0.5rem 0;'>€700-€1,500</p>
                <p style='font-size: 0.85rem; margin: 0;'>Professional work</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class='price-point premium'>
                <h4>Premium</h4>
                <p style='font-size: 1.5rem; margin: 0.5rem 0;'>€1,500+</p>
                <p style='font-size: 0.85rem; margin: 0;'>High-end computing</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("""
        <div class='comparison-box'>
            <h3>🎯 Key Price Factors</h3>
            <p><strong>Brand Influence:</strong> Apple and premium brands typically command 20-30% higher prices</p>
            <p><strong>GPU Impact:</strong> Dedicated graphics can add €200-€800 to the price</p>
            <p><strong>Storage Type:</strong> SSDs cost more but provide 5-10x faster performance than HDDs</p>
            <p><strong>Display Quality:</strong> High-resolution displays with IPS/Retina can add €100-€300</p>
        </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
        <p style='text-align: center; color: #6c757d; font-size: 0.95rem;'>
            🤖 Powered by Advanced Machine Learning | Built by Arnav
        </p>
        <p style='text-align: center; color: #9ca3af; font-size: 0.85rem; margin-top: -0.5rem;'>
            Predictions based on market data and specifications analysis
        </p>
    """, unsafe_allow_html=True)