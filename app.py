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

# Header
st.title("💻 Laptop Price Predictor")
st.write("Instant price estimates powered by advanced machine learning algorithms")

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
st.sidebar.header("Configure Your Laptop")

# Brand first
brand = st.sidebar.selectbox("Brand", sorted(df['Company'].dropna().unique()))

# Then filter models for that brand
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
    st.header("Your Laptop Configuration")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Brand")
        st.write(brand)
        st.subheader("Type")
        st.write(typename)
    with col2:
        st.subheader("CPU")
        st.write(cpu_company)
        st.subheader("GPU")
        st.write(gpu_company)
    with col3:
        st.subheader("RAM")
        st.write(f"{ram} GB")
        st.subheader("Storage")
        st.write(f"{total_storage} GB")
    st.markdown("**Detailed Specifications**")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Model:** {selected_model}")
        st.write(f"**Operating System:** {os}")
        st.write(f"**CPU Frequency:** {cpu_freq} GHz")
        st.write(f"**Primary Storage:** {storage_type}")
        st.write(f"**Display PPI:** {ppi}")
    with col2:
        st.write(f"**GPU Model:** {gpu_model}")
        st.write(f"**Secondary Storage:** {secondary_storage}")
        st.write(f"**Weight:** {weight} kg")
        st.write(f"**Touchscreen:** {touchscreen}")
        st.write(f"**IPS Panel:** {ips_panel} | **Retina:** {retina}")

with tab2:
    st.header("Get Your Price Estimate")
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
    if st.button("🔮 Calculate Price Estimate"):
        with st.spinner("🤖 analyzing your configuration..."):
            time.sleep(1.5)
            predicted_log = model.predict(input_data)[0]
            predicted_price = np.expm1(predicted_log)
            st.subheader("Estimated Laptop Price")
            st.write(f"**€{predicted_price:,.2f}**")
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
            st.write(f"{icon} **{category} Laptop**")
            st.write(description)
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
                        margin=dict(l=100, r=50, t=70, b=50),
                    )
                    st.plotly_chart(fig, use_container_width=True)
            # Additional insights
            st.subheader("💡 Smart Recommendations")
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
                    st.write(rec)
            else:
                st.write("✅ Your configuration is well-balanced!")

with tab3:
    st.header("Market Price Analysis")
    st.subheader("📊 Price Range Guide")
    st.write("Understanding where your laptop fits in the market")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Budget**")
        st.write("€300-€700")
        st.write("Basic tasks & browsing")
    with col2:
        st.markdown("**Mid-Range**")
        st.write("€700-€1,500")
        st.write("Professional work")
    with col3:
        st.markdown("**Premium**")
        st.write("€1,500+")
        st.write("High-end computing")
    st.markdown("---")
    st.subheader("🎯 Key Price Factors")
    st.write("- **Brand Influence:** Apple and premium brands typically command 20-30% higher prices")
    st.write("- **GPU Impact:** Dedicated graphics can add €200-€800 to the price")
    st.write("- **Storage Type:** SSDs cost more but provide 5-10x faster performance than HDDs")
    st.write("- **Display Quality:** High-resolution displays with IPS/Retina can add €100-€300")

# Footer
st.markdown("---")
st.write("🤖 Powered by Advanced Machine Learning | Built by Arnav")
st.write("Predictions based on market data and specifications analysis")