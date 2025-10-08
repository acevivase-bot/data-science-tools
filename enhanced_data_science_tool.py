import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import uuid
import io
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ============= PAGE CONFIG =============
st.set_page_config(
    page_title="Enhanced Data Science Tool",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============= ENHANCED CSS STYLING =============
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e, #2ca02c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        padding: 1rem;
    }

    .sub-header {
        font-size: 1.3rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
        font-style: italic;
    }

    .session-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        text-align: center;
        font-family: 'Monaco', monospace;
        font-weight: bold;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }

    .step-indicator {
        display: flex;
        justify-content: space-between;
        margin: 2rem 0;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }

    .step {
        flex: 1;
        text-align: center;
        padding: 1rem;
        border-radius: 8px;
        margin: 0 0.5rem;
        transition: all 0.3s ease;
    }

    .step-active {
        background: #1f77b4;
        color: white;
        transform: scale(1.05);
        box-shadow: 0 4px 15px rgba(31, 119, 180, 0.3);
    }

    .step-completed {
        background: #2ca02c;
        color: white;
    }

    .step-inactive {
        background: #e9ecef;
        color: #6c757d;
    }

    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }

    .metric-number {
        font-size: 2.5rem;
        font-weight: bold;
        display: block;
    }

    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }

    .success-msg {
        background: linear-gradient(90deg, #d4edda, #c3e6cb);
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #2ca02c;
    }

    .warning-msg {
        background: linear-gradient(90deg, #fff3cd, #ffeaa7);
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #ff7f0e;
    }

    .error-msg {
        background: linear-gradient(90deg, #f8d7da, #f1c2c7);
        border: 1px solid #f1c2c7;
        color: #721c24;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #d62728;
    }

    .info-msg {
        background: linear-gradient(90deg, #cce7f0, #b8daff);
        border: 1px solid #b8daff;
        color: #004085;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #17becf;
    }

    .upload-area {
        background: linear-gradient(45deg, #f8f9fa, #e9ecef);
        border: 2px dashed #1f77b4;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }

    .upload-area:hover {
        background: linear-gradient(45deg, #e9ecef, #dee2e6);
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }

    .footer {
        background: #343a40;
        color: white;
        padding: 2rem;
        border-radius: 5px;
        text-align: center;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# ============= SESSION MANAGEMENT =============
def initialize_session():
    """Initialize session dengan unique ID dan data persistence"""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8].upper()

    # Initialize data containers
    if 'original_data' not in st.session_state:
        st.session_state.original_data = None
    if 'current_data' not in st.session_state:
        st.session_state.current_data = None
    if 'data_history' not in st.session_state:
        st.session_state.data_history = []
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 'Getting Data'
    if 'completed_steps' not in st.session_state:
        st.session_state.completed_steps = []

def update_step(step_name):
    """Update current step dan mark sebagai completed"""
    if st.session_state.current_step not in st.session_state.completed_steps:
        st.session_state.completed_steps.append(st.session_state.current_step)
    st.session_state.current_step = step_name

def save_data_state(operation_name):
    """Save current data state untuk history tracking"""
    if st.session_state.current_data is not None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.data_history.append({
            'operation': operation_name,
            'timestamp': timestamp,
            'shape': st.session_state.current_data.shape,
            'columns': list(st.session_state.current_data.columns)
        })

# ============= PROGRESS INDICATOR =============
def show_progress_indicator():
    """Show step-by-step progress indicator"""
    steps = ['Getting Data', 'Processing Data', 'Visualize Data', 'Feature Engineering', 'Model Building']

    progress_html = '<div class="step-indicator">'

    for i, step in enumerate(steps):
        if step in st.session_state.completed_steps:
            class_name = "step step-completed"
            icon = "✅"
        elif step == st.session_state.current_step:
            class_name = "step step-active"
            icon = "🔄"
        else:
            class_name = "step step-inactive"
            icon = "⭕"

        progress_html += f'<div class="{class_name}"><div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{icon}</div><div style="font-weight: bold;">{step}</div></div>'

    progress_html += '</div>'
    st.markdown(progress_html, unsafe_allow_html=True)

# ============= DATA LOADING FUNCTIONS =============
def load_data_section():
    """Enhanced data loading dengan multiple sources"""
    st.markdown("## 📤 Data Loading")

    # Data source selection
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### 📊 Data Sources")
        data_source = st.selectbox(
            "Choose data source:",
            ["📁 Upload File", "🌐 Load Sample Data"],
            key=f"data_source_{st.session_state.session_id}"
        )

    with col2:
        if data_source == "📁 Upload File":
            st.markdown("""
            <div class="upload-area">
                <h3>📁 Upload Your Dataset</h3>
                <p>Drag and drop your file here or click to browse</p>
                <small>Supported formats: CSV, Excel (XLSX), JSON | Max size: 200MB</small>
            </div>
            """, unsafe_allow_html=True)

            uploaded_file = st.file_uploader(
                "Choose a file",
                type=['csv', 'xlsx', 'json'],
                key=f"file_uploader_{st.session_state.session_id}"
            )

            if uploaded_file is not None:
                try:
                    with st.spinner("🔄 Loading data..."):
                        # Load based on file type
                        if uploaded_file.name.endswith('.csv'):
                            data = pd.read_csv(uploaded_file)
                        elif uploaded_file.name.endswith('.xlsx'):
                            data = pd.read_excel(uploaded_file)
                        elif uploaded_file.name.endswith('.json'):
                            data = pd.read_json(uploaded_file)

                        # Store in session state
                        st.session_state.original_data = data.copy()
                        st.session_state.current_data = data.copy()

                        # Save initial state
                        save_data_state("Data Loaded")

                        st.markdown(f'<div class="success-msg">✅ Successfully loaded {uploaded_file.name}!</div>', unsafe_allow_html=True)

                        # Show basic info
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.markdown(f'<div class="metric-card"><span class="metric-number">{data.shape[0]:,}</span><span class="metric-label">Rows</span></div>', unsafe_allow_html=True)
                        with col2:
                            st.markdown(f'<div class="metric-card"><span class="metric-number">{data.shape[1]}</span><span class="metric-label">Columns</span></div>', unsafe_allow_html=True)
                        with col3:
                            memory_usage = data.memory_usage(deep=True).sum() / 1024**2
                            st.markdown(f'<div class="metric-card"><span class="metric-number">{memory_usage:.1f}</span><span class="metric-label">MB</span></div>', unsafe_allow_html=True)
                        with col4:
                            missing_pct = (data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100
                            st.markdown(f'<div class="metric-card"><span class="metric-number">{missing_pct:.1f}%</span><span class="metric-label">Missing</span></div>', unsafe_allow_html=True)

                except Exception as e:
                    st.markdown(f'<div class="error-msg">❌ Error loading file: {str(e)}</div>', unsafe_allow_html=True)

        elif data_source == "🌐 Load Sample Data":
            st.markdown("### 📊 Sample Datasets")

            sample_choice = st.selectbox(
                "Choose a sample dataset:",
                ["Titanic", "Iris", "Tips", "Car Crashes"],
                key=f"sample_choice_{st.session_state.session_id}"
            )

            if st.button("🔄 Load Sample Data", key=f"load_sample_{st.session_state.session_id}"):
                try:
                    if sample_choice == "Titanic":
                        data = sns.load_dataset('titanic')
                    elif sample_choice == "Iris":
                        data = sns.load_dataset('iris')
                    elif sample_choice == "Tips":
                        data = sns.load_dataset('tips')
                    elif sample_choice == "Car Crashes":
                        data = sns.load_dataset('car_crashes')

                    st.session_state.original_data = data.copy()
                    st.session_state.current_data = data.copy()
                    save_data_state(f"Sample Data: {sample_choice}")

                    st.markdown(f'<div class="success-msg">✅ {sample_choice} dataset loaded successfully!</div>', unsafe_allow_html=True)
                    st.rerun()

                except Exception as e:
                    st.markdown(f'<div class="error-msg">❌ Error loading sample data: {str(e)}</div>', unsafe_allow_html=True)

    # Data preview if loaded
    if st.session_state.current_data is not None:
        st.markdown("### 👀 Data Preview")
        st.dataframe(st.session_state.current_data.head(), use_container_width=True, height=200)

def processing_data_section():
    """Enhanced data processing dengan persistence"""
    st.markdown("## 🔧 Data Processing")

    if st.session_state.current_data is None:
        st.markdown('<div class="warning-msg">⚠️ No data loaded. Please go to "Getting Data" section first.</div>', unsafe_allow_html=True)
        return

    # Processing options tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview", "🧹 Clean Data", "🔍 Missing Values", "💾 Export"
    ])

    with tab1:
        st.markdown("### 📊 Dataset Overview")

        # Current data metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="metric-card"><span class="metric-number">{st.session_state.current_data.shape[0]:,}</span><span class="metric-label">Total Rows</span></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><span class="metric-number">{st.session_state.current_data.shape[1]}</span><span class="metric-label">Total Columns</span></div>', unsafe_allow_html=True)
        with col3:
            duplicates = st.session_state.current_data.duplicated().sum()
            st.markdown(f'<div class="metric-card"><span class="metric-number">{duplicates:,}</span><span class="metric-label">Duplicates</span></div>', unsafe_allow_html=True)
        with col4:
            missing = st.session_state.current_data.isnull().sum().sum()
            st.markdown(f'<div class="metric-card"><span class="metric-number">{missing:,}</span><span class="metric-label">Missing Values</span></div>', unsafe_allow_html=True)

        # Data preview
        st.markdown("### 🔍 Data Preview")
        st.dataframe(st.session_state.current_data.head(10), use_container_width=True)

        # Data types
        st.markdown("### 🔢 Data Types")
        dtypes_df = pd.DataFrame({
            'Column': st.session_state.current_data.columns,
            'Data Type': st.session_state.current_data.dtypes,
            'Non-Null Count': st.session_state.current_data.count(),
            'Null Count': st.session_state.current_data.isnull().sum()
        }).reset_index(drop=True)
        st.dataframe(dtypes_df, use_container_width=True)

    with tab2:
        st.markdown("### 🧹 Data Cleaning")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🗑️ Remove Operations")

            # Remove duplicates
            if st.button("🔄 Remove Duplicate Rows", key=f"remove_dupes_{st.session_state.session_id}"):
                original_shape = st.session_state.current_data.shape
                st.session_state.current_data = st.session_state.current_data.drop_duplicates()
                new_shape = st.session_state.current_data.shape
                removed = original_shape[0] - new_shape[0]

                save_data_state("Duplicates Removed")
                st.success(f"✅ Removed {removed} duplicate rows!")
                st.rerun()

            # Remove columns
            cols_to_remove = st.multiselect(
                "Select columns to remove:",
                st.session_state.current_data.columns,
                key=f"cols_remove_{st.session_state.session_id}"
            )

            if cols_to_remove and st.button("🗑️ Remove Selected Columns", key=f"remove_cols_{st.session_state.session_id}"):
                st.session_state.current_data = st.session_state.current_data.drop(columns=cols_to_remove)
                save_data_state("Columns Removed")
                st.success(f"✅ Removed {len(cols_to_remove)} columns!")
                st.rerun()

        with col2:
            st.markdown("#### ✂️ Filter Operations")

            # Filter by column value
            filter_col = st.selectbox(
                "Select column to filter:",
                st.session_state.current_data.columns,
                key=f"filter_col_{st.session_state.session_id}"
            )

            if filter_col:
                if st.session_state.current_data[filter_col].dtype in ['int64', 'float64']:
                    # Numeric filter
                    min_val = float(st.session_state.current_data[filter_col].min())
                    max_val = float(st.session_state.current_data[filter_col].max())

                    filter_range = st.slider(
                        f"Filter {filter_col} range:",
                        min_val, max_val, (min_val, max_val),
                        key=f"filter_range_{st.session_state.session_id}"
                    )

                    if st.button("🎯 Apply Numeric Filter", key=f"apply_filter_{st.session_state.session_id}"):
                        mask = (st.session_state.current_data[filter_col] >= filter_range[0]) & (st.session_state.current_data[filter_col] <= filter_range[1])
                        original_rows = len(st.session_state.current_data)
                        st.session_state.current_data = st.session_state.current_data[mask]
                        new_rows = len(st.session_state.current_data)

                        save_data_state("Numeric Filter Applied")
                        st.success(f"✅ Filtered data: {original_rows} → {new_rows} rows")
                        st.rerun()

    with tab3:
        st.markdown("### 🔍 Missing Values Handling")

        missing_summary = st.session_state.current_data.isnull().sum()
        missing_cols = missing_summary[missing_summary > 0]

        if len(missing_cols) == 0:
            st.markdown('<div class="success-msg">🎉 No missing values found in the dataset!</div>', unsafe_allow_html=True)
        else:
            # Show missing values chart
            fig = px.bar(
                x=missing_cols.index,
                y=missing_cols.values,
                title="Missing Values by Column",
                labels={'x': 'Column', 'y': 'Missing Count'}
            )
            st.plotly_chart(fig, use_container_width=True)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 🛠️ Fill Missing Values")

                col_to_fill = st.selectbox(
                    "Select column to fill:",
                    missing_cols.index,
                    key=f"fill_col_{st.session_state.session_id}"
                )

                if col_to_fill:
                    if st.session_state.current_data[col_to_fill].dtype in ['int64', 'float64']:
                        fill_method = st.selectbox(
                            "Fill method for numeric column:",
                            ["Mean", "Median", "Mode"],
                            key=f"fill_method_{st.session_state.session_id}"
                        )

                        if st.button("🔧 Fill Missing Values", key=f"fill_missing_{st.session_state.session_id}"):
                            if fill_method == "Mean":
                                fill_val = st.session_state.current_data[col_to_fill].mean()
                            elif fill_method == "Median":
                                fill_val = st.session_state.current_data[col_to_fill].median()
                            elif fill_method == "Mode":
                                fill_val = st.session_state.current_data[col_to_fill].mode().iloc[0]

                            missing_count = st.session_state.current_data[col_to_fill].isnull().sum()
                            st.session_state.current_data[col_to_fill] = st.session_state.current_data[col_to_fill].fillna(fill_val)
                            save_data_state(f"Missing Values Filled: {fill_method}")
                            st.success(f"✅ Filled {missing_count} missing values with {fill_method}!")
                            st.rerun()

            with col2:
                st.markdown("#### 🗑️ Remove Missing Values")

                # Remove rows with any missing values
                if st.button("🗑️ Remove Rows with Missing Values", key=f"remove_missing_{st.session_state.session_id}"):
                    original_rows = len(st.session_state.current_data)
                    st.session_state.current_data = st.session_state.current_data.dropna()
                    new_rows = len(st.session_state.current_data)
                    removed = original_rows - new_rows

                    save_data_state("Rows with Missing Values Removed")
                    st.success(f"✅ Removed {removed} rows with missing values!")
                    st.rerun()

    with tab4:
        st.markdown("### 💾 Export Processed Data")

        if st.session_state.current_data is not None:
            col1, col2, col3 = st.columns(3)

            with col1:
                # CSV download
                csv = st.session_state.current_data.to_csv(index=False)
                st.download_button(
                    label="📄 Download as CSV",
                    data=csv,
                    file_name=f"processed_data_{st.session_state.session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key=f"csv_download_{st.session_state.session_id}",
                    use_container_width=True
                )

            with col2:
                # Excel download
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    st.session_state.current_data.to_excel(writer, sheet_name='Processed_Data', index=False)

                excel_data = output.getvalue()
                st.download_button(
                    label="📊 Download as Excel",
                    data=excel_data,
                    file_name=f"processed_data_{st.session_state.session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"excel_download_{st.session_state.session_id}",
                    use_container_width=True
                )

            with col3:
                # JSON download
                json_data = st.session_state.current_data.to_json(orient='records', indent=2)
                st.download_button(
                    label="🔗 Download as JSON",
                    data=json_data,
                    file_name=f"processed_data_{st.session_state.session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    key=f"json_download_{st.session_state.session_id}",
                    use_container_width=True
                )

def visualize_data_section():
    """Enhanced data visualization dengan persistent state"""
    st.markdown("## 📊 Data Visualization")

    if st.session_state.current_data is None:
        st.markdown('<div class="warning-msg">⚠️ No data loaded. Please go to "Getting Data" section first.</div>', unsafe_allow_html=True)
        return

    # Visualization tabs
    tab1, tab2, tab3 = st.tabs([
        "📈 Basic Charts", "🎨 Advanced Charts", "🤖 Auto EDA"
    ])

    with tab1:
        st.markdown("### 📈 Basic Visualizations")

        col1, col2 = st.columns([1, 2])

        with col1:
            chart_type = st.selectbox(
                "Select chart type:",
                ["Histogram", "Bar Chart", "Line Chart", "Scatter Plot", "Box Plot", "Pie Chart"],
                key=f"chart_type_{st.session_state.session_id}"
            )

            # Column selection based on chart type
            if chart_type in ["Histogram", "Box Plot"]:
                numeric_cols = st.session_state.current_data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    selected_col = st.selectbox("Select column:", numeric_cols, key=f"viz_col_{st.session_state.session_id}")
                else:
                    st.error("No numeric columns available for this chart type.")
                    return

            elif chart_type in ["Bar Chart", "Pie Chart"]:
                categorical_cols = st.session_state.current_data.select_dtypes(include=['object']).columns
                if len(categorical_cols) > 0:
                    selected_col = st.selectbox("Select column:", categorical_cols, key=f"viz_col_{st.session_state.session_id}")
                else:
                    st.error("No categorical columns available for this chart type.")
                    return

            elif chart_type in ["Scatter Plot", "Line Chart"]:
                numeric_cols = st.session_state.current_data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) >= 2:
                    x_col = st.selectbox("Select X column:", numeric_cols, key=f"x_col_{st.session_state.session_id}")
                    y_col = st.selectbox("Select Y column:", [col for col in numeric_cols if col != x_col], key=f"y_col_{st.session_state.session_id}")
                else:
                    st.error("Need at least 2 numeric columns for this chart type.")
                    return

        with col2:
            try:
                if chart_type == "Histogram":
                    fig = px.histogram(st.session_state.current_data, x=selected_col, title=f"Histogram of {selected_col}")

                elif chart_type == "Bar Chart":
                    value_counts = st.session_state.current_data[selected_col].value_counts().head(10)
                    fig = px.bar(x=value_counts.index, y=value_counts.values, 
                               title=f"Top 10 Values - {selected_col}",
                               labels={'x': selected_col, 'y': 'Count'})

                elif chart_type == "Line Chart":
                    fig = px.line(st.session_state.current_data, x=x_col, y=y_col,
                                title=f"Line Chart: {y_col} vs {x_col}")

                elif chart_type == "Scatter Plot":
                    fig = px.scatter(st.session_state.current_data, x=x_col, y=y_col,
                                   title=f"Scatter Plot: {y_col} vs {x_col}")

                elif chart_type == "Box Plot":
                    fig = px.box(st.session_state.current_data, y=selected_col, title=f"Box Plot of {selected_col}")

                elif chart_type == "Pie Chart":
                    value_counts = st.session_state.current_data[selected_col].value_counts().head(8)
                    fig = px.pie(values=value_counts.values, names=value_counts.index,
                               title=f"Distribution of {selected_col}")

                # Update layout
                fig.update_layout(
                    font=dict(size=12),
                    title_font_size=16,
                    template='plotly_white'
                )

                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error creating visualization: {str(e)}")

    with tab2:
        st.markdown("### 🎨 Advanced Visualizations")

        # Correlation heatmap
        numeric_cols = st.session_state.current_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            st.markdown("#### 🔗 Correlation Matrix")
            corr_matrix = st.session_state.current_data[numeric_cols].corr()
            fig = px.imshow(corr_matrix, 
                          labels=dict(color="Correlation"),
                          title="Correlation Heatmap")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Need at least 2 numeric columns for correlation heatmap.")

    with tab3:
        st.markdown("### 🤖 Automated EDA")

        if st.button("🚀 Generate Auto EDA Report", key=f"auto_eda_{st.session_state.session_id}"):
            with st.spinner("Generating automated EDA report..."):
                # Basic dataset overview
                st.markdown("#### 📊 Dataset Overview")

                overview_cols = st.columns(4)
                with overview_cols[0]:
                    st.metric("Rows", f"{st.session_state.current_data.shape[0]:,}")
                with overview_cols[1]:
                    st.metric("Columns", st.session_state.current_data.shape[1])
                with overview_cols[2]:
                    missing_pct = (st.session_state.current_data.isnull().sum().sum() / (st.session_state.current_data.shape[0] * st.session_state.current_data.shape[1])) * 100
                    st.metric("Missing %", f"{missing_pct:.1f}%")
                with overview_cols[3]:
                    duplicates = st.session_state.current_data.duplicated().sum()
                    st.metric("Duplicates", f"{duplicates:,}")

def feature_engineering_section():
    """Enhanced feature engineering"""
    st.markdown("## ⚙️ Feature Engineering")

    if st.session_state.current_data is None:
        st.markdown('<div class="warning-msg">⚠️ No data loaded. Please go to "Getting Data" section first.</div>', unsafe_allow_html=True)
        return

    # Feature engineering tabs
    tab1, tab2 = st.tabs(["🔧 Basic Operations", "📊 Encoding"])

    with tab1:
        st.markdown("### 🔧 Basic Feature Operations")

        # Create new column from existing columns
        numeric_cols = st.session_state.current_data.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) >= 2:
            col1, col2, col3 = st.columns(3)

            with col1:
                col1_select = st.selectbox("First column:", numeric_cols, key=f"feat_col1_{st.session_state.session_id}")
            with col2:
                operation = st.selectbox("Operation:", ["+", "-", "*", "/"], key=f"feat_op_{st.session_state.session_id}")
            with col3:
                col2_select = st.selectbox("Second column:", [col for col in numeric_cols if col != col1_select], key=f"feat_col2_{st.session_state.session_id}")

            new_col_name = st.text_input("New column name:", value=f"{col1_select}_{operation}_{col2_select}", key=f"new_col_name_{st.session_state.session_id}")

            if st.button("➕ Create Feature", key=f"create_feat_{st.session_state.session_id}"):
                try:
                    if operation == "+":
                        st.session_state.current_data[new_col_name] = st.session_state.current_data[col1_select] + st.session_state.current_data[col2_select]
                    elif operation == "-":
                        st.session_state.current_data[new_col_name] = st.session_state.current_data[col1_select] - st.session_state.current_data[col2_select]
                    elif operation == "*":
                        st.session_state.current_data[new_col_name] = st.session_state.current_data[col1_select] * st.session_state.current_data[col2_select]
                    elif operation == "/":
                        st.session_state.current_data[new_col_name] = st.session_state.current_data[col1_select] / st.session_state.current_data[col2_select]

                    save_data_state(f"Feature Created: {new_col_name}")
                    st.success(f"✅ Created feature: {new_col_name}")
                    st.rerun()

                except Exception as e:
                    st.error(f"Error creating feature: {str(e)}")

    with tab2:
        st.markdown("### 📊 Categorical Encoding")

        categorical_cols = st.session_state.current_data.select_dtypes(include=['object']).columns

        if len(categorical_cols) > 0:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 🏷️ Label Encoding")

                label_col = st.selectbox(
                    "Select column for label encoding:",
                    categorical_cols,
                    key=f"label_col_{st.session_state.session_id}"
                )

                if st.button("🏷️ Apply Label Encoding", key=f"label_encode_{st.session_state.session_id}"):
                    try:
                        le = LabelEncoder()
                        st.session_state.current_data[f"{label_col}_encoded"] = le.fit_transform(st.session_state.current_data[label_col])

                        save_data_state("Label Encoding Applied")
                        st.success(f"✅ Applied label encoding to {label_col}")
                        st.rerun()

                    except Exception as e:
                        st.error(f"Error in label encoding: {str(e)}")

            with col2:
                st.markdown("#### 🎯 One-Hot Encoding")

                onehot_col = st.selectbox(
                    "Select column for one-hot encoding:",
                    categorical_cols,
                    key=f"onehot_col_{st.session_state.session_id}"
                )

                if st.button("🎯 Apply One-Hot Encoding", key=f"onehot_encode_{st.session_state.session_id}"):
                    try:
                        unique_count = st.session_state.current_data[onehot_col].nunique()

                        if unique_count > 10:
                            st.warning(f"Column has {unique_count} unique values. This might create many columns.")

                        # Create dummy variables
                        dummies = pd.get_dummies(st.session_state.current_data[onehot_col], prefix=onehot_col)

                        # Add to dataframe
                        st.session_state.current_data = pd.concat([st.session_state.current_data, dummies], axis=1)

                        save_data_state("One-Hot Encoding Applied")
                        st.success(f"✅ Applied one-hot encoding to {onehot_col} ({unique_count} new columns created)")
                        st.rerun()

                    except Exception as e:
                        st.error(f"Error in one-hot encoding: {str(e)}")
        else:
            st.info("No categorical columns available for encoding.")

def model_building_section():
    """Enhanced model building dengan persistent state"""
    st.markdown("## 🤖 Model Building")

    if st.session_state.current_data is None:
        st.markdown('<div class="warning-msg">⚠️ No data loaded. Please go to "Getting Data" section first.</div>', unsafe_allow_html=True)
        return

    # Model building tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Setup", "🏗️ Train Models", "📊 Evaluate"])

    with tab1:
        st.markdown("### 🎯 Model Setup")

        # Problem type selection
        col1, col2 = st.columns(2)

        with col1:
            problem_type = st.selectbox(
                "Select problem type:",
                ["Classification", "Regression"],
                key=f"problem_type_{st.session_state.session_id}"
            )

        with col2:
            test_size = st.slider(
                "Test set size:",
                0.1, 0.5, 0.2, 0.05,
                key=f"test_size_{st.session_state.session_id}"
            )

        # Target variable selection
        target_col = st.selectbox(
            "Select target variable:",
            st.session_state.current_data.columns,
            key=f"target_col_{st.session_state.session_id}"
        )

        # Feature selection
        available_features = [col for col in st.session_state.current_data.columns if col != target_col]
        selected_features = st.multiselect(
            "Select features:",
            available_features,
            default=available_features,
            key=f"features_{st.session_state.session_id}"
        )

        if st.button("🎯 Setup Model Training", key=f"setup_model_{st.session_state.session_id}"):
            try:
                # Prepare features and target
                X = st.session_state.current_data[selected_features]
                y = st.session_state.current_data[target_col]

                # Handle missing values
                X = X.fillna(X.mean() if X.select_dtypes(include=[np.number]).shape[1] > 0 else X.mode().iloc[0])

                # Encode categorical features
                for col in X.select_dtypes(include=['object']).columns:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))

                # Encode target if classification
                if problem_type == "Classification" and not pd.api.types.is_numeric_dtype(y):
                    le_target = LabelEncoder()
                    y = le_target.fit_transform(y.astype(str))
                    st.session_state.target_encoder = le_target

                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

                # Store in session state
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.problem_type = problem_type

                st.success("✅ Model setup completed!")
                st.write(f"Training set: {X_train.shape[0]} samples")
                st.write(f"Test set: {X_test.shape[0]} samples")
                st.write(f"Features: {len(selected_features)}")

            except Exception as e:
                st.error(f"Error in model setup: {str(e)}")

    with tab2:
        st.markdown("### 🏗️ Train Models")

        if not all(key in st.session_state for key in ['X_train', 'X_test', 'y_train', 'y_test']):
            st.warning("Please complete model setup first!")
            return

        # Model selection
        if st.session_state.problem_type == "Classification":
            model_options = ["Logistic Regression", "Random Forest Classifier"]
        else:
            model_options = ["Linear Regression", "Random Forest Regressor"]

        selected_model = st.selectbox(
            "Select model to train:",
            model_options,
            key=f"selected_model_{st.session_state.session_id}"
        )

        if st.button("🚀 Train Model", key=f"train_model_{st.session_state.session_id}"):
            try:
                with st.spinner("Training model..."):
                    # Initialize model
                    if selected_model == "Logistic Regression":
                        model = LogisticRegression(random_state=42)
                    elif selected_model == "Linear Regression":
                        model = LinearRegression()
                    elif selected_model == "Random Forest Classifier":
                        model = RandomForestClassifier(n_estimators=100, random_state=42)
                    elif selected_model == "Random Forest Regressor":
                        model = RandomForestRegressor(n_estimators=100, random_state=42)

                    # Train model
                    model.fit(st.session_state.X_train, st.session_state.y_train)

                    # Make predictions
                    y_pred = model.predict(st.session_state.X_test)

                    # Store model and predictions
                    st.session_state.trained_model = model
                    st.session_state.y_pred = y_pred
                    st.session_state.model_name = selected_model

                    st.success(f"✅ {selected_model} trained successfully!")

                    # Show basic metrics
                    if st.session_state.problem_type == "Classification":
                        accuracy = accuracy_score(st.session_state.y_test, y_pred)
                        st.metric("Accuracy", f"{accuracy:.4f}")
                    else:
                        mse = mean_squared_error(st.session_state.y_test, y_pred)
                        r2 = r2_score(st.session_state.y_test, y_pred)
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("MSE", f"{mse:.4f}")
                        with col2:
                            st.metric("R² Score", f"{r2:.4f}")

            except Exception as e:
                st.error(f"Error training model: {str(e)}")

    with tab3:
        st.markdown("### 📊 Model Evaluation")

        if 'trained_model' not in st.session_state:
            st.warning("Please train a model first!")
            return

        # Performance metrics
        if st.session_state.problem_type == "Classification":
            accuracy = accuracy_score(st.session_state.y_test, st.session_state.y_pred)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Model", st.session_state.model_name)
                st.metric("Accuracy", f"{accuracy:.4f}")

            with col2:
                # Prediction distribution
                pred_counts = pd.Series(st.session_state.y_pred).value_counts()
                fig = px.bar(x=pred_counts.index, y=pred_counts.values, 
                           title="Prediction Distribution")
                st.plotly_chart(fig, use_container_width=True)

        else:  # Regression
            mse = mean_squared_error(st.session_state.y_test, st.session_state.y_pred)
            r2 = r2_score(st.session_state.y_test, st.session_state.y_pred)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Model", st.session_state.model_name)
                st.metric("MSE", f"{mse:.4f}")
                st.metric("R² Score", f"{r2:.4f}")

            with col2:
                # Prediction vs Actual plot
                fig = px.scatter(
                    x=st.session_state.y_test, 
                    y=st.session_state.y_pred,
                    title="Predicted vs Actual Values",
                    labels={'x': 'Actual Values', 'y': 'Predicted Values'}
                )
                st.plotly_chart(fig, use_container_width=True)

# ============= MAIN APPLICATION =============
def main():
    # Initialize session
    initialize_session()

    # Header
    st.markdown('<h1 class="main-header">🔬 Enhanced Data Science Tool</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Professional data analysis platform with persistent state management</p>', unsafe_allow_html=True)

    # Session badge
    st.markdown(f'<div class="session-badge">🔒 Session: {st.session_state.session_id} | Multi-User Ready</div>', unsafe_allow_html=True)

    # Progress indicator
    show_progress_indicator()

    # Sidebar navigation
    st.sidebar.markdown("### 🧭 Navigation")
    selected_section = st.sidebar.selectbox(
        "Choose section:",
        ['Getting Data', 'Processing Data', 'Visualize Data', 'Feature Engineering', 'Model Building'],
        index=['Getting Data', 'Processing Data', 'Visualize Data', 'Feature Engineering', 'Model Building'].index(st.session_state.current_step),
        key=f"navigation_{st.session_state.session_id}"
    )

    # Update current step
    if selected_section != st.session_state.current_step:
        update_step(selected_section)

    # Data status sidebar
    st.sidebar.markdown("### 📊 Data Status")
    if st.session_state.current_data is not None:
        st.sidebar.success("✅ Data Loaded")
        st.sidebar.write(f"📏 Shape: {st.session_state.current_data.shape}")
        if st.session_state.data_history:
            st.sidebar.write(f"🔄 Operations: {len(st.session_state.data_history)}")
    else:
        st.sidebar.warning("⚠️ No Data Loaded")

    # Main content based on selected section
    if selected_section == 'Getting Data':
        load_data_section()
    elif selected_section == 'Processing Data':
        processing_data_section()
    elif selected_section == 'Visualize Data':
        visualize_data_section()
    elif selected_section == 'Feature Engineering':
        feature_engineering_section()
    elif selected_section == 'Model Building':
        model_building_section()

    # Footer
    st.markdown("""
    <div class="footer">
        <h4>🔬 Enhanced Data Science Tool</h4>
        <p>Created by Vito Devara | Phone: 081259795994</p>
        <p>Persistent data state • Session management • Professional UI</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
