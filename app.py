from pathlib import Path
import json
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from tensorflow import keras

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='Customer Churn Intelligence | ANN',page_icon='🧠',layout='wide')
THEMES = {
    'Aurora':('#0B1020','#171F38','#E9F0FF','#8B5CF6','#22D3EE'),
    'Ocean':('#071B2C','#102F46','#E8F7FF','#0284C7','#2DD4BF'),
    'Sunset':('#241425','#3B203A','#FFF1F2','#F97316','#EC4899'),
    'Emerald':('#071F1A','#13392E','#EDFFF7','#10B981','#A3E635'),
    'Royal':('#15112C','#2A2250','#F3EFFF','#A78BFA','#F472B6'),
    'Light':('#F4F7FC','#FFFFFF','#17243B','#2563EB','#0D9488')
}
with st.sidebar:
    st.title('🧠 Churn Intelligence')
    page = st.radio('Navigate',['Overview','Customer Analytics','Predict Churn','Model Performance'])
    theme = st.selectbox('Color theme',list(THEMES))
    st.caption('ANN-powered customer retention analytics')
bg,panel,fg,accent,accent2 = THEMES[theme]
st.markdown(f'''<style>
.stApp {{background:{bg};color:{fg};}}
[data-testid="stSidebar"] {{background:{panel};}}
h1,h2,h3,p,label {{color:{fg};}}
[data-testid="stMetric"] {{background:{panel};border:1px solid {accent}55;padding:16px;border-radius:16px;}}
.stButton>button[kind="primary"] {{background:linear-gradient(100deg,{accent},{accent2});color:white;border:0;border-radius:12px;}}
</style>''',unsafe_allow_html=True)
@st.cache_data
def load_data(): return pd.read_csv(ROOT/'data'/'Churn_Modelling.csv')
@st.cache_resource
def load_artifacts():
    model = keras.models.load_model(ROOT/'models'/'churn_ann.keras')
    prep = joblib.load(ROOT/'models'/'preprocessor.joblib')
    return model,prep

df = load_data()
st.title('Customer Churn Intelligence')
st.caption('Interactive analytics and customer churn prediction using an Artificial Neural Network')
if page == 'Overview':
    a,b,c,d = st.columns(4)
    a.metric('Total Customers',f'{len(df):,}')
    b.metric('Churned',f'{df.Exited.sum():,}')
    c.metric('Churn Rate',f'{df.Exited.mean():.2%}')
    d.metric('Retention Rate',f'{1-df.Exited.mean():.2%}')
    left,right = st.columns(2)
    with left: st.plotly_chart(px.pie(df,names='Exited',title='Customer Churn Distribution',hole=.55,color_discrete_sequence=[accent,accent2]),use_container_width=True)
    with right: st.plotly_chart(px.histogram(df,x='Geography',color='Exited',barmode='group',title='Churn by Geography',color_discrete_sequence=[accent,accent2]),use_container_width=True)
elif page == 'Customer Analytics':
    geography = st.multiselect('Geography',sorted(df.Geography.unique()),default=sorted(df.Geography.unique()))
    filtered = df[df.Geography.isin(geography)]
    if filtered.empty: st.warning('Select at least one geography.')
    else:
        x,y = st.columns(2)
        with x: st.plotly_chart(px.histogram(filtered,x='Age',color='Exited',nbins=25,title='Age and Churn',color_discrete_sequence=[accent,accent2]),use_container_width=True)
        with y: st.plotly_chart(px.histogram(filtered,x='NumOfProducts',color='Exited',barmode='group',title='Products and Churn',color_discrete_sequence=[accent,accent2]),use_container_width=True)
        st.dataframe(filtered.drop(columns=['Surname','CustomerId','RowNumber']),use_container_width=True)
elif page == 'Predict Churn':
    st.subheader('Customer Risk Assessment')
    with st.form('predict'):
        a,b,c = st.columns(3)
        with a:
            score=st.slider('Credit score',300,850,650)
            geo=st.selectbox('Geography',sorted(df.Geography.unique()))
            gender=st.selectbox('Gender',sorted(df.Gender.unique()))
            age=st.slider('Age',18,100,40)
        with b:
            tenure=st.slider('Tenure (years)',0,10,5)
            balance=st.number_input('Account balance',min_value=0.,value=50000.,step=1000.)
            products=st.slider('Number of products',1,4,1)
        with c:
            card=st.selectbox('Has credit card',['Yes','No'])
            active=st.selectbox('Active member',['Yes','No'])
            salary=st.number_input('Estimated salary',min_value=0.,value=100000.,step=1000.)
        submit=st.form_submit_button('Predict Customer Churn',type='primary')
    if submit:
        try:
            model,prep=load_artifacts()
            sample=pd.DataFrame([{'CreditScore':score,'Geography':geo,'Gender':gender,'Age':age,'Tenure':tenure,'Balance':balance,'NumOfProducts':products,'HasCrCard':int(card=='Yes'),'IsActiveMember':int(active=='Yes'),'EstimatedSalary':salary}])
            probability=float(model.predict(prep.transform(sample).astype('float32'),verbose=0)[0][0])
            threshold=json.loads((ROOT/'models'/'metrics.json').read_text())['threshold']
            st.metric('Estimated Churn Probability',f'{probability:.1%}')
            st.progress(probability)
            if probability>=threshold: st.error('Higher predicted churn risk — prioritize customer retention outreach.')
            else: st.success('Lower predicted churn risk at the selected decision threshold.')
            st.caption('Prediction is a statistical estimate, not a certainty. Use it as decision support.')
        except FileNotFoundError: st.error('Model artifacts missing. Run: python train_model.py')
else:
    st.subheader('Held-out Test Performance')
    try:
        metrics=json.loads((ROOT/'models'/'metrics.json').read_text())
        cols=st.columns(5)
        for col,key in zip(cols,['accuracy','precision','recall','f1','roc_auc']): col.metric(key.upper().replace('_','-'),f'{metrics[key]:.2%}')
        st.caption(f"Test samples: {metrics['test_samples']:,} · Decision threshold: {metrics['threshold']:.2f} (chosen using validation data)")
    except FileNotFoundError: st.info('Run python train_model.py to generate performance metrics.')
