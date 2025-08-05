import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import statsmodels.api as sm




# ---------- Custom Styling (Blue Theme) ----------
st.markdown(
    """
  <style>
    .stApp {
        background-color: #fce4ec;  /* Light Pink */
    }
    .stSidebar {
        background-color: #f8bbd0;  /* Sidebar Pink */
    }
</style>
    """,
    unsafe_allow_html=True
)



# ---------- Header ----------
st.markdown(
    """
    <div style='display: flex; justify-content: center; align-items: center; flex-direction: column;'>
        <img src='https://upload.wikimedia.org/wikipedia/commons/8/83/TUMS_Signature_Variation_1_BLUE.png' width='200' style='margin-bottom: 10px;'/>
    </div>
    """,
    unsafe_allow_html=True
)


st.title('🤖🧬 Machine Learning APP for Predicting Female Infertility Risk')
st.info('Predict the **Female Infertility** status in Women based on health data using ML models(XGBoost).')










# ---------- Load and Preprocess Data ----------
@st.cache_data
def load_data():
    url = "https://github.com/Bahsobi/SIRI_infertility-/raw/refs/heads/main/Filtered_NHANES_cleaned_no_missing_data_final.xlsx"
    df = pd.read_excel(url)


    
    # Mapping categorical values
    df['Race_Ethnicity'] = df['Race_Ethnicity'].map({
        1: "Mexican American",
        2: "Other Hispanic",
        3: "Non-Hispanic White",
        4: "Non-Hispanic Black",
        6: "Non-Hispanic Asian",
        7: "Other Race - Including Multi-Racial"
    })

    df['Education_Level'] = df['Education_Level'].map({
        1: "Less than 9th grade",
        2: "9-11th grade",
        3: "High school graduate or equivalent",
        4: "Some college or AA degree",
        5: "College graduate or above"
    })

    df['Hypertension'] = df['Hypertension'].map({
        0: "No",
        1: "Yes"
    })

    df['Diabetes_Status'] = df['Diabetes_Status'].map({
        1: "Diabetes",
        2: "Healthy",  
    })

    return df

df = load_data()

#################################################

# ---------- Features ----------
target = 'Female_infertility'

categorical_features = ['Race_Ethnicity', 'Education_Level', 'Hypertension', 'Diabetes_Status']
numerical_features = ['Age', 'BMI', 'Total_Cholesterol', 'SIRI']
features = categorical_features + numerical_features


X = df[features]
y = df[target]

# ---------- Preprocessing ----------
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ('num', StandardScaler(), numerical_features)
])

# ---------- XGBoost Pipeline ----------
model = Pipeline([
    ('prep', preprocessor),
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
model.fit(X_train, y_train)

# ---------- Feature Importance ----------
xgb_model = model.named_steps['xgb']
encoder = model.named_steps['prep'].named_transformers_['cat']
feature_names = encoder.get_feature_names_out(categorical_features).tolist() + numerical_features
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': xgb_model.feature_importances_
}).sort_values(by='Importance', ascending=False)


# ---------- Logistic Regression for Odds Ratio ----------
odds_pipeline = Pipeline([
    ('prep', preprocessor),
    ('logreg', LogisticRegression(max_iter=1000))
])
odds_pipeline.fit(X_train, y_train)
log_model = odds_pipeline.named_steps['logreg']
odds_df = pd.DataFrame({
    'Feature': feature_names,
    'Odds Ratio': np.exp(log_model.coef_[0])
}).sort_values(by='Odds Ratio', ascending=False)

















################################################################################
# ---------- Sidebar Input ----------
st.sidebar.header("📝 Input Data")



#part1
# Fixed Category Options (Based on Your Divisions)



race_options = df['Race_Ethnicity'].dropna().unique().tolist()
edu_options = df['Education_Level'].dropna().unique().tolist()
htn_options = df['Hypertension'].dropna().unique().tolist()
dm_options = df['Diabetes_Status'].dropna().unique().tolist()



#part2
# Numerical Inputs (Fixed Range like previous style)



siri = st.sidebar.number_input("SIRI (0.1 - 10.0)", min_value=0.1, max_value=10.0, value=1.0)

age = st.sidebar.number_input("Age (18 - 80)", min_value=18, max_value=80, value=30)
bmi = st.sidebar.number_input("BMI (14.6 - 82.0)", min_value=14.6, max_value=82.0, value=25.0)
total_cholesterol = st.sidebar.number_input("Total Cholesterol (80 - 400)", min_value=80.0, max_value=400.0, value=200.0)



#part3
# Categorical Inputs with New Divisions

race = st.sidebar.selectbox("Race/Ethnicity", race_options)
edu = st.sidebar.selectbox("Education Level", edu_options)
htn = st.sidebar.selectbox("Hypertension", htn_options)
dm = st.sidebar.selectbox("Diabetes Status", dm_options)



user_input = pd.DataFrame([{
    'Age': age,
    'BMI': bmi,
    'Total_Cholesterol': total_cholesterol,
    'SIRI': siri,
    'Race_Ethnicity': race,
    'Education_Level': edu,
    'Hypertension': htn,
    'Diabetes_Status': dm
}])








#برای تغییر یا حذف یک متغیر باید هم در مپینگ و هم در فیچر تارگت اولیه و هم در ساید بار تغییر بدی
##################################








# ---------- Prediction ----------
prediction = model.predict(user_input)[0]
probability = model.predict_proba(user_input)[0][1]
odds_value = probability / (1 - probability)

# ---------- Display Result ----------
if prediction == 1:
    st.error(f"""
        ⚠️ **Prediction: At Risk of Female Infertility**

        🧮 **Probability:** {probability:.2%}  
        🎲 **Odds:** {odds_value:.2f}
    """)
else:
    st.success(f"""
        ✅ **Prediction: Not at Risk of Female Infertility**

        🧮 **Probability:** {probability:.2%}  
        🎲 **Odds:** {odds_value:.2f}
    """)

# ---------- Show Tables ----------
st.subheader("📊 Odds Ratios for Female Infertility (Logistic Regression)")
st.dataframe(odds_df)

st.subheader("💡 Feature Importances (XGBoost)")
st.dataframe(importance_df)

# ---------- Plot Feature Importances ----------
st.subheader("📈 Bar Chart: Feature Importances")
fig, ax = plt.subplots()
sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
st.pyplot(fig)

# ---------- Quartile Odds Ratio for SIRI ----------
st.subheader("📉 Odds Ratios for Female Infertility by SIRI Quartiles")
df_siri = df[['SIRI', target]].copy()
df_siri['SIRI_quartile'] = pd.qcut(df_siri['SIRI'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

X_q = pd.get_dummies(df_siri['SIRI_quartile'], drop_first=True)
X_q = sm.add_constant(X_q).astype(float)
y_q = df_siri[target].astype(float)

model_q = sm.Logit(y_q, X_q).fit(disp=False)
ors = np.exp(model_q.params)
ci = model_q.conf_int()
ci.columns = ['2.5%', '97.5%']
ci = np.exp(ci)

or_df = pd.DataFrame({
    'Quartile': ors.index,
    'Odds Ratio': ors.values,
    'CI Lower': ci['2.5%'],
    'CI Upper': ci['97.5%'],
    'p-value': model_q.pvalues
}).query("Quartile != 'const'")

st.dataframe(or_df.set_index('Quartile').style.format("{:.2f}"))

fig3, ax3 = plt.subplots()
sns.pointplot(data=or_df, x='Quartile', y='Odds Ratio', join=False, capsize=0.2, errwidth=1.5)
ax3.axhline(1, linestyle='--', color='gray')
ax3.set_title("Odds Ratios for Female Infertility by SIRI Quartiles")
st.pyplot(fig3)

# ---------- Summary ----------
with st.expander("📋 Data Summary"):
    st.write(df.describe())

st.subheader("🎯 Female Infertility Distribution")
fig2, ax2 = plt.subplots()
df[target].value_counts().plot.pie(
    autopct='%1.1f%%', labels=['No Infertility', 'Infertility'], ax=ax2, colors=["#81c784", "#e57373"])
ax2.set_ylabel("")
st.pyplot(fig2)

with st.expander("🔍 Sample Data (First 10 Rows)"):
    st.dataframe(df.head(10))
