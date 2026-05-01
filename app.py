import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu

# ---------- 1. ASSET LOADING ----------
@st.cache_resource
def load_assets():
    # Loading the brain and the scaler
    m = joblib.load("model.pkl")
    s = joblib.load("scaler.pkl")
    return m, s

@st.cache_data
def load_csv():
    # Loading the data for graphs
    return pd.read_csv("train.csv")

try:
    model, scaler = load_assets()
    df = load_csv()
except Exception as e:
    st.error(f"⚠️ Files missing! Error: {e}")
    st.info("Make sure 'train.csv' and the 'model/' folder are in your project directory.")
    st.stop()

# ---------- 2. PAGE STYLE ----------
st.set_page_config(page_title="House Price Analysis", layout="wide")

st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .card { background-color: white; padding: 20px; border-radius: 10px; 
            box-shadow: 0px 4px 10px rgba(0,0,0,0.05); text-align: center; border-top: 5px solid #1E88E5; }
    .card h3 { color: #555; font-size: 16px; }
    .card p { font-size: 24px; font-weight: bold; color: #1E88E5; }
</style>
""", unsafe_allow_html=True)

# ---------- 3. NAVIGATION (4 TABS) ----------
selected = option_menu(
    menu_title=None,
    options=["Home", "Data Analysis","Trend Analysis", "Price Prediction"],
    icons=["house", "info-circle", "bar-chart-line", "cash-stack"],
    default_index=0,
    orientation="horizontal",
    styles={"nav-link-selected": {"background-color": "#1E88E5"}}
)

# ---------- 4. TAB 1: HOME ----------
if selected == "Home":
    st.markdown("<h1 style='text-align:center;'>🏠 House Price Trends Analysis</h1>", unsafe_allow_html=True)
    st.write("---")
    # Overview Section
    st.subheader("📌 Project Overview")

    st.write("""
    This dashboard provides comprehensive analysis of real estate data and predicts
    property prices using a Machine Learning model (XGBoost).

    It helps users understand:
    • Market price trends  
    • Feature influence on house prices  
    • Location-based pricing differences  
    • Real-time property valuation  
    """)
    st.write("")
    # How It Works Section
    st.subheader("⚙️ How It Works")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Step 1 — Data Analysis**
        - Explore dataset statistics  
        - Visualize correlations  
        - Identify key price drivers  
        """)

    with col2:
        st.markdown("""
        **Step 2 — Machine Learning**
        - XGBoost Algorithm
        - Feature scaling applied  
        - Model trained & evaluated  
        - Real-time price prediction  
        """)

    st.write("")

     # Technology Section
    st.subheader("🛠 Technologies Used")

    st.markdown("""
    - Python  
    - Pandas & NumPy  
    - Matplotlib & Seaborn  
    - Plotly  
    - Scikit-learn  
    - Streamlit  
    - XGBoost
    - streamlit-option-menu
    - joblib
   
    """)

    st.write("")

    st.title("🤖 Model Information")
    col_a, col_b = st.columns(2)
    with col_a:
        st.info("#### Algorithm: XGBoost")
        st.write("XGBoost is a high-performance Gradient Boosting algorithm used for regression.")
    with col_b:
        st.success("#### Training Details")
        st.write("- **Features Used:** 20\n- **Estimators:** 1000\n- **Scaler:** StandardScaler")


    st.info("Use the navigation menu above to explore Data Analysis, Trends, and Price Prediction.")

    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(f'<div class="card"><h3>Avg Price</h3><p>${int(df["SalePrice"].mean()):,}</p></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="card"><h3>Max Price</h3><p>${int(df["SalePrice"].max()):,}</p></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="card"><h3>Total Houses</h3><p>{len(df):,}</p></div>', unsafe_allow_html=True)
    st.subheader("Property Data Preview")
# --- 4 . DATA ANALYSIS (TAB 2) ---
elif selected == "Data Analysis":
    st.title("📊 Statistical Data Visualizations")
    st.subheader("Dataset Preview (First 10 Records)")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.subheader("Statistical Summary")
    st.write("This table shows mean, minimum, maximum and spread of numerical features.")
    st.dataframe(df.describe())

    # --- HEATMAP SECTION ---
    st.subheader("Feature Correlation Heatmap")
    st.write("This heatmap visualizes the 'Service Layer' logic by showing which features drive the Sale Price.")

    # Use the 'df' already loaded at the top of the script
    numeric_df = df.select_dtypes(include=['int64', 'float64'])

    # Only show the top 10 most correlated features to keep the 5x3 clean
    top_corr_features = numeric_df.corr()['SalePrice'].sort_values(ascending=False).head(10).index
    subset_corr = numeric_df[top_corr_features].corr()

    # Your centered layout
    left, center, right = st.columns([1, 2, 1])

    with center:
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.heatmap(
            subset_corr, 
            annot=True, 
            cmap="coolwarm", 
            linewidths=0.5, 
            fmt=".2f", 
            annot_kws={"size": 6}, 
            ax=ax
        )
        ax.set_title("Correlation Matrix (Top Drivers)", fontsize=10)
        plt.xticks(fontsize=7, rotation=45)
        plt.yticks(fontsize=7)
        st.pyplot(fig)


# ---------- 5. TAB 3: Trend Analysis ----------
elif selected == "Trend Analysis":
    st.title("📈 Market Trend Analysis")
    st.write("Visualizing the key factors that drive the Ames housing market.")

    # --- ROW 1: PRE-EXISTING GRAPHS (Keeping your originals) ---
    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Living Area vs Price")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=df, x="GrLivArea", y="SalePrice", alpha=0.5, color="#1E88E5", ax=ax1)
        st.pyplot(fig1)
    with col_r:
        st.subheader("Quality Rating Impact")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.barplot(data=df, x="OverallQual", y="SalePrice", hue="OverallQual", palette="Blues_d", legend=False, ax=ax2)
        st.pyplot(fig2)

    st.markdown("---")

    # --- ROW 2: BEDROOMS & PRICE DISTRIBUTION ---
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Bedrooms vs Price")
        # Using a Boxplot to show the price range for each bedroom count
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df, x="BedroomAbvGr", y="SalePrice", palette="viridis", ax=ax3)
        ax3.set_xlabel("Number of Bedrooms")
        st.pyplot(fig3)

    with col4:
        st.subheader("Price Distribution (Histogram)")
        # Showing how prices are spread across the dataset
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        sns.histplot(df["SalePrice"], kde=True, color="#1E88E5", ax=ax4)
        ax4.set_title(f"Mean Price: ${int(df['SalePrice'].mean()):,}")
        st.pyplot(fig4)

    # --- ROW 3: LOCATION (NEIGHBORHOOD) VS PRICE ---
    st.markdown("---")
    st.subheader("Location-Based Pricing (Neighborhoods)")
    st.write("Top 10 most expensive vs least expensive neighborhoods.")
    
    # Calculate average price per neighborhood
    neigh_price = df.groupby("Neighborhood")["SalePrice"].mean().sort_values(ascending=False).reset_index()
    
    fig5, ax5 = plt.subplots(figsize=(12, 5))
    sns.barplot(data=neigh_price, x="Neighborhood", y="SalePrice", palette="magma", ax=ax5)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    st.pyplot(fig5)



# ---------- 7. TAB 4: PRICE PREDICTION ----------
elif selected == "Price Prediction":
    st.title("💰 Smart Price Estimator")
    st.write("Enter house details to get an instant valuation from the XGBoost model.")
    
    # 1. Inputs Section with Increased Max Values
    ca, cb = st.columns(2)
    with ca:
        qual = st.slider('Overall Quality (1-10)', 1, 10, 7)
        area = st.number_input('Living Area (SqFt)', 500, 20000, 2500) # Increased to 20k SqFt
        garage = st.selectbox('Garage Capacity (Cars)', [0, 1, 2, 3, 4, 5], index=2)
        bath = st.radio('Full Bathrooms', [1, 2, 3, 4, 5], horizontal=True, index=2)
    with cb:
        bsmt = st.number_input('Total Basement (SqFt)', 0, 10000, 1200) # Increased to 10k SqFt
        year = st.slider('Year Built', 1880, 2024, 2010) # Updated to modern years
        fireplaces = st.slider('Fireplaces', 0, 5, 1)
        lot = st.number_input('Lot Area (SqFt)', 1000, 100000, 12000) # Increased to 100k SqFt

    if st.button("Generate Prediction", use_container_width=True):
        # Feature names must match exactly what the scaler expects
        feature_names = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', 
                         '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd', 
                         'GarageYrBlt', 'MasVnrArea', 'Fireplaces', 'BsmtFinSF1', 'LotFrontage', 
                         'WoodDeckSF', '2ndFlrSF', 'OpenPorchSF', 'HalfBath', 'LotArea']
        
        # Mapping inputs to the 20 required features
        inputs = [
            qual, area, garage, (garage * 250), bsmt, (area * 0.6), bath, 
            (bath + 3), year, year, year, 0, fireplaces, 
            400, 70, 0, 0, 40, 0, lot
        ]
        
        # 1. Prediction Logic
        input_df = pd.DataFrame([inputs], columns=feature_names)
        scaled_data = scaler.transform(input_df)
        pred_usd = model.predict(scaled_data)[0]
        
        # 2. Currency Conversion Logic (1 USD = 83 INR)
        conversion_rate = 83.0
        pred_inr = pred_usd * conversion_rate
        
        st.markdown("---")
        
        # 3. Display Results in Two Columns (USD and INR)
        res1, res2 = st.columns(2)
        
        with res1:
            st.metric("Estimated Price (USD)", f"${pred_usd:,.2f}")
            st.caption("Based on Ames Housing Market Rates")
            
        with res2:
            # Formatting for Lakhs/Crores
            if pred_inr >= 10000000:
                inr_display = f"₹{pred_inr/10000000:.2f} Crores"
            else:
                inr_display = f"₹{pred_inr/100000:.2f} Lakhs"
                
            st.metric("Estimated Price (INR)", inr_display)
            st.caption(f"Conversion Rate: 1 USD = ₹{conversion_rate}")

        st.success("✅ Prediction generated successfully by XGBoost Service Layer.")
