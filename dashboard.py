import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

def load_data():
    try:
        import mysql.connector
        db = mysql.connector.connect(
            host="localhost", user="root", password="zbhya6a4", database="hrdb", use_pure=True
        )
        df = pd.read_sql('SELECT * FROM table1', db).merge(
            pd.read_sql('SELECT * FROM table2', db), on="EmployeeNumber", how="left"
        )
        db.close()
        return df
    except Exception as e:
        st.warning(f"Could not load from database: {e}. Loading from CSV.")
        return pd.read_csv('data.csv')

data = load_data()

st.set_page_config(layout="wide", page_title="HR Attrition Dashboard", page_icon="📊", initial_sidebar_state="expanded")
with st.container():
    st.markdown("""
        <style>
        .main, .block-container {background-color: #fff !important; color: #222 !important;}
        .block-container { max-width: 1200px; margin: auto; padding-top: 1rem; padding-bottom: 1rem; }
        </style>
        """, unsafe_allow_html=True)

st.title('HR Attrition Dashboard')
st.markdown('---')

# --- SIDEBAR FILTERS ---
departments = data['Department'].dropna().unique()
selected_departments = st.sidebar.multiselect('Filter by Department', options=departments, default=list(departments))
filtered_data = data[data['Department'].isin(selected_departments)]
jobroles = filtered_data['JobRole'].dropna().unique()
selected_jobroles = st.sidebar.multiselect('Filter by Job Role', options=jobroles, default=list(jobroles))
filtered_data = filtered_data[filtered_data['JobRole'].isin(selected_jobroles)]

# 1. Attrition rate by departments
st.header('Attrition Rate by Department')
attrition_pct = (
    filtered_data.groupby('Department')['Attrition']
    .apply(lambda x: (x == 'Yes').mean() * 100 if x.dtype == object else x.mean() * 100)
    .reset_index()
)
company_avg = attrition_pct['Attrition'].mean()
x_vals = attrition_pct['Department'].tolist()
fig1 = go.Figure([
    go.Bar(x=x_vals, y=attrition_pct['Attrition'], name='Attrition Rate (%)', marker_color='cornflowerblue')
])
if x_vals:
    fig1.add_shape(
        type="line", x0=-0.5, x1=len(x_vals)-0.5, y0=company_avg, y1=company_avg,
        line=dict(color="crimson", width=3, dash="dash"), xref="x", yref="y", layer="above"
    )
    fig1.add_trace(go.Scatter(
        x=[x_vals[0], x_vals[-1]], y=[company_avg]*2, mode='lines', name='Company Avg',
        line=dict(color='crimson', dash='dash'), showlegend=True, hoverinfo='skip', visible='legendonly'
    ))
fig1.update_layout(
    yaxis_title='Attrition Rate (%)', xaxis_title='Department',
    title='Attrition Rate by Department (with Company Average)',
    barmode='group', legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    width=900, height=400, margin=dict(l=40, r=40, t=60, b=40)
)
st.plotly_chart(fig1, use_container_width=False)

# 3. Attrition percentage by job role in the sales department
st.header('Attrition Percentage by Job Role in Sales Department')
sales_attrition = filtered_data[filtered_data['Department'] == 'Sales']
if not sales_attrition.empty:
    attrition_counts = sales_attrition.groupby(['JobRole', 'Attrition']).size().reset_index(name='Count')
    attrition_counts['Percentage'] = attrition_counts['Count'] / attrition_counts.groupby('JobRole')['Count'].transform('sum') * 100
    fig3 = px.bar(
        attrition_counts, x='JobRole', y='Percentage', color='Attrition', barmode='group',
        labels={'Percentage':'Attrition %'}, title='Attrition % by Job Role (Sales)'
    )
    fig3.update_layout(width=900, height=400, margin=dict(l=40, r=40, t=60, b=40))
    st.plotly_chart(fig3, use_container_width=False)

# 4. Years at company of sales representatives
st.header('Years at Company of Sales Representatives')
sales_reps = filtered_data[filtered_data['JobRole'] == 'Sales Representative'].copy()
if not sales_reps.empty:
    bins = [0, 1, 3, 5, 10, sales_reps['YearsAtCompany'].max() + 1]
    labels = ['<1', '1-2', '3-4', '5-9', '10+']
    sales_reps['YearsAtCompanyGroup'] = pd.cut(sales_reps['YearsAtCompany'], bins=bins, labels=labels, right=False)
    years_counts = sales_reps['YearsAtCompanyGroup'].value_counts().sort_index()
    attrition_rates = sales_reps.groupby('YearsAtCompanyGroup')['Attrition'].apply(lambda x: (x == 'Yes').mean() * 100).sort_index()
    fig4 = go.Figure([go.Bar(
        x=years_counts.index.astype(str), y=years_counts.values, name='Number of Employees', marker_color='#4F8BB8'
    )])
    for label in years_counts.index:
        rate = attrition_rates.get(label, None)
        if rate is not None:
            fig4.add_annotation(
                x=label, y=years_counts[label] + max(years_counts.values) * 0.02,
                text=f"{rate:.1f}%", showarrow=False, font=dict(size=12, color='black', family='Arial')
            )
    fig4.update_layout(
        xaxis_title='Years at Company', yaxis_title='Number of Employees',
        title='Years at Company of Sales Representatives (Attrition Rate Above Each Bar)',
        width=900, height=400, margin=dict(l=40, r=40, t=60, b=40)
    )
    st.plotly_chart(fig4, use_container_width=False)

# 5. Education level of sales representatives
st.header('Education Level of Sales Representatives')
education_labels = {1: 'Below College', 2: 'College', 3: 'Bachelor', 4: 'Master', 5: 'Doctor'}
if not sales_reps.empty:
    education_counts = sales_reps['Education'].value_counts().sort_index()
    education_counts.index = education_counts.index.map(lambda x: education_labels.get(x, str(x)))
    attrition_rates_edu = sales_reps.groupby('Education')['Attrition'].apply(lambda x: (x == 'Yes').mean() * 100).sort_index()
    attrition_rates_edu.index = attrition_rates_edu.index.map(lambda x: education_labels.get(x, str(x)))
    fig5 = go.Figure([go.Bar(
        x=education_counts.index, y=education_counts.values, marker_color='mediumseagreen', name='Number of Employees'
    )])
    for label in education_counts.index:
        rate = attrition_rates_edu.get(label, None)
        if rate is not None:
            fig5.add_annotation(
                x=label, y=education_counts[label] + max(education_counts.values) * 0.02,
                text=f"{rate:.1f}%", showarrow=False, font=dict(size=12, color='black', family='Arial')
            )
    fig5.update_layout(
        xaxis_title='Education Level', yaxis_title='Number of Employees',
        title='Education Level of Sales Representatives (Attrition Rate Above Each Bar)',
        width=900, height=400, margin=dict(l=40, r=40, t=60, b=40)
    )
    st.plotly_chart(fig5, use_container_width=False)

# 2. Job satisfaction distribution in each department by attrition (as styled table)
st.header('Job Satisfaction Distribution by Attrition')
if selected_departments:
    cols = st.columns(len(selected_departments))
    for idx, dept in enumerate(selected_departments):
        dept_data = filtered_data[filtered_data['Department'] == dept]
        if dept_data.empty: continue
        job_sat_table = pd.crosstab(
            dept_data['JobSatisfaction'], dept_data['Attrition'], normalize='index'
        ).mul(100).round(2)
        if {'No', 'Yes'}.issubset(job_sat_table.columns):
            job_sat_table = job_sat_table[['No', 'Yes']]
        with cols[idx]:
            st.subheader(f"{dept}")
            st.dataframe(
                job_sat_table.style.format("{:.2f}%").background_gradient(cmap='YlGnBu', axis=None)
                .set_caption(f'Job Satisfaction by Attrition (%) - {dept}'),
                use_container_width=True
            )

# --- Distance From Home Histogram (All Employees) ---
st.header('Distance From Home Distribution (All Employees)')
if not filtered_data.empty and 'DistanceFromHome' in filtered_data.columns:
    num_bins = 10
    hist, bin_edges = np.histogram(filtered_data['DistanceFromHome'], bins=num_bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_indices = np.digitize(filtered_data['DistanceFromHome'], bin_edges, right=False)
    avg_job_satisfaction = [
        filtered_data[bin_indices == i]['JobSatisfaction'].mean() if (bin_indices == i).any() else None
        for i in range(1, len(bin_edges))
    ]
    attrition_rate = [
        (filtered_data[bin_indices == i]['Attrition'] == 'Yes').mean() * 100 if (bin_indices == i).any() else None
        for i in range(1, len(bin_edges))
    ]
    fig_dist = go.Figure([go.Bar(
        x=bin_centers,
        y=hist,
        width=(bin_edges[1] - bin_edges[0]),
        marker_color='skyblue',
        name='Employees',
        marker_line_color='black',
        marker_line_width=1
    )])
    for i, (x, y) in enumerate(zip(bin_centers, hist)):
        y_offset = max(hist) * 0.02
        if avg_job_satisfaction[i] is not None:
            fig_dist.add_annotation(
                x=x, y=y + y_offset, 
                text=f"<b>{avg_job_satisfaction[i]:.2f}</b>",
                showarrow=False, 
                font=dict(size=11, color='darkred'),
                align='center'
            )
            y_offset += max(hist) * 0.05
        if attrition_rate[i] is not None:
            fig_dist.add_annotation(
                x=x, y=y + y_offset, 
                text=f"<b>{attrition_rate[i]:.1f}%</b>",
                showarrow=False, 
                font=dict(size=11, color='green'),
                align='center'
            )
            
    fig_dist.update_layout(
        title='Distance From Home Histogram (All Employees)',
        xaxis_title='Distance From Home', yaxis_title='Number of Employees',
        width=900, height=400, margin=dict(l=40, r=40, t=60, b=40)
    )
    st.plotly_chart(fig_dist, use_container_width=False)