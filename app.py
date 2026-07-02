import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

from prediction import predict_churn


# Load data and model

df = pd.read_csv("European_Bank.csv")

import os
import joblib

print("Current working directory:", os.getcwd())
print("Loading from:", os.path.abspath("bank_churn_pipeline.pkl"))

model = joblib.load("bank_churn_pipeline.pkl")
print("Model loaded successfully!")


# Page setup

st.set_page_config(
    page_title="Bank Churn Predictor",
    layout="wide"
)


# Title

st.title(
    "🏦 Bank Customer Churn Prediction Dashboard"
)

st.write(
    "Machine Learning based customer churn prediction system"
)

st.divider()


# ---------------- INPUT ----------------


col1, col2 = st.columns(2)


with col1:

    CreditScore = st.number_input(
        "Credit Score",
        300,
        900,
        650
    )


    Geography = st.selectbox(
        "Geography",
        ["France","Germany","Spain"]
    )


    Gender = st.selectbox(
        "Gender",
        ["Male","Female"]
    )


    Age = st.number_input(
        "Age",
        18,
        100,
        35
    )


    Tenure = st.number_input(
        "Tenure",
        0,
        10,
        5
    )



with col2:


    Balance = st.number_input(
        "Balance",
        value=50000.0
    )


    NumOfProducts = st.number_input(
        "Number of Products",
        1,
        4,
        1
    )


    HasCrCard = st.selectbox(
        "Has Credit Card",
        [0,1]
    )


    IsActiveMember = st.selectbox(
        "Active Member",
        [0,1]
    )


    EstimatedSalary = st.number_input(
        "Estimated Salary",
        value=60000.0
    )


    Year = st.number_input(
        "Year",
        2000,
        2030,
        2026
    )



# ---------------- PREDICT ----------------


if st.button("🔮 Predict Churn"):


    data = pd.DataFrame({


        "CreditScore":[CreditScore],

        "Geography":[Geography],

        "Gender":[Gender],

        "Age":[Age],

        "Tenure":[Tenure],

        "Balance":[Balance],

        "NumOfProducts":[NumOfProducts],

        "HasCrCard":[HasCrCard],

        "IsActiveMember":[IsActiveMember],

        "EstimatedSalary":[EstimatedSalary],

        "Year":[Year],


        "BalanceSalaryRatio":[
            Balance/EstimatedSalary
        ],


        "ProductDensity":[
            NumOfProducts/(Tenure+1)
        ],


        "EngagementScore":[
            HasCrCard+IsActiveMember
        ],


        "AgeTenureInteraction":[
            Age*Tenure
        ]

    })


    prediction = predict_churn(
        model,
        data
    )



    # Result


    c1,c2 = st.columns(2)


    with c1:

        st.metric(
            "Churn Probability",
            f"{prediction*100:.2f}%"
        )



    with c2:


        if prediction < 0.3:

            st.success(
                "🟢 Low Risk Customer"
            )


        elif prediction < 0.7:

            st.warning(
                "🟡 Medium Risk Customer"
            )


        else:

            st.error(
                "🔴 High Risk Customer"
            )



    st.divider()



    # Gauge


    st.subheader(
        "🎯 Churn Probability Gauge"
    )


    gauge = go.Figure(

        go.Indicator(

            mode="gauge+number",

            value=prediction*100,

            title={
                "text":"Churn Risk %"
            },

            gauge={
                "axis":{
                    "range":[0,100]
                }
            }

        )

    )


    st.plotly_chart(
        gauge,
        use_container_width=True
    )



    # Risk Graph


    st.subheader(
        "📊 Risk Category"
    )


    risk_df = pd.DataFrame({

        "Category":[
            "Churn Risk",
            "Safe Customer"
        ],


        "Percentage":[
            prediction*100,
            (1-prediction)*100
        ]

    })


    risk_fig = px.bar(

        risk_df,

        x="Category",

        y="Percentage",

        text="Percentage",

        title="Customer Risk Percentage"

    )


    st.plotly_chart(
        risk_fig,
        use_container_width=True
    )



    # Customer Profile


    st.subheader(
        "📊 Customer Profile Analysis"
    )


    profile_df = pd.DataFrame({

        "Feature":[
            "Credit Score",
            "Age",
            "Balance",
            "Tenure",
            "Products",
            "Salary"
        ],


        "Value":[

            CreditScore,
            Age,
            Balance,
            Tenure,
            NumOfProducts,
            EstimatedSalary

        ]

    })


    profile_graph = px.bar(

        profile_df,

        x="Feature",

        y="Value",

        text="Value"

    )


    st.plotly_chart(
        profile_graph,
        use_container_width=True
    )



    # Explanation


    st.subheader(
        "📝 Prediction Explanation"
    )


    if prediction > 0.5:

        st.write(
            "This customer has higher churn probability based on the given profile."
        )

    else:

        st.write(
            "This customer has lower churn probability and appears stable."
        )



# Dataset graph (always visible)

st.divider()

st.subheader(
    "📈 Overall Customer Churn Distribution"
)


fig, ax = plt.subplots(figsize=(6,4))


sns.countplot(
    x="Exited",
    data=df,
    ax=ax
)


ax.set_title(
    "Customer Churn Distribution"
)


st.pyplot(fig)

# ---------------- CUSTOMER ACTIVITY VS CHURN ----------------

st.divider()

st.subheader("👤 Customer Activity vs Churn")


activity_df = df.groupby(
    ["IsActiveMember", "Exited"]
).size().reset_index(name="Count")


activity_graph = px.bar(

    activity_df,

    x="IsActiveMember",

    y="Count",

    color="Exited",

    barmode="group",

    text="Count",

    title="Customer Activity vs Churn"

)


activity_graph.update_layout(

    xaxis_title="Active Member (0 = No, 1 = Yes)",

    yaxis_title="Number of Customers"

)


st.plotly_chart(

    activity_graph,

    use_container_width=True

)

# ---------------- CHURN BY GEOGRAPHY ----------------

st.divider()

st.subheader("🌍 Churn by Geography")


geo_df = df.groupby(
    ["Geography", "Exited"]
).size().reset_index(name="Count")


geo_graph = px.bar(

    geo_df,

    x="Geography",

    y="Count",

    color="Exited",

    barmode="group",

    text="Count",

    title="Churn by Geography"

)


geo_graph.update_layout(

    xaxis_title="Country",

    yaxis_title="Number of Customers",

    legend_title="Churn Status"

)


st.plotly_chart(

    geo_graph,

    use_container_width=True

)


# ---------------- CHURN BY GENDER ----------------

st.divider()

st.subheader("👥 Churn by Gender")


gender_df = df.groupby(
    ["Gender", "Exited"]
).size().reset_index(name="Count")


gender_graph = px.bar(

    gender_df,

    x="Gender",

    y="Count",

    color="Exited",

    barmode="group",

    text="Count",

    title="Churn by Gender"

)


gender_graph.update_layout(

    xaxis_title="Gender",

    yaxis_title="Number of Customers",

    legend_title="Churn Status"

)


st.plotly_chart(

    gender_graph,

    use_container_width=True

)

import numpy as np
from scipy.stats import gaussian_kde


st.subheader("💰 Balance Distribution with Density Curve")


fig = go.Figure()


for status in [0,1]:

    data = df[df["Exited"]==status]["Balance"]

    kde = gaussian_kde(data)


    x_values = np.linspace(
        data.min(),
        data.max(),
        200
    )


    fig.add_trace(

        go.Scatter(

            x=x_values,

            y=kde(x_values),

            mode="lines",

            name=f"Exited {status}"

        )

    )


fig.update_layout(

    title="Balance Distribution by Churn Status",

    xaxis_title="Account Balance",

    yaxis_title="Density"

)


st.plotly_chart(

    fig,

    use_container_width=True

)

# ---------------- CORRELATION HEATMAP ----------------

st.divider()

st.subheader("🔥 Feature Correlation Heatmap")


# Select only numeric columns
corr = df.corr(numeric_only=True)


heatmap = px.imshow(

    corr,

    text_auto=True,

    aspect="auto",

    title="Feature Correlation Heatmap",

    color_continuous_scale="RdBu"

)


heatmap.update_layout(

    xaxis_title="Features",

    yaxis_title="Features",

    height=700

)


st.plotly_chart(

    heatmap,

    use_container_width=True

)
