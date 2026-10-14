from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tensorflow import keras

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / 'models'
MODEL_DIR.mkdir(exist_ok=True)
df = pd.read_csv(ROOT / 'data' / 'Churn_Modelling.csv')
features = ['CreditScore','Geography','Gender','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary']
X = df[features]
y = df['Exited']
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)
cat = ['Geography', 'Gender']
num = [c for c in features if c not in cat]
prep = ColumnTransformer([('numeric',StandardScaler(),num),('category',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cat)])
A = prep.fit_transform(X_train).astype('float32')
B = prep.transform(X_val).astype('float32')
C = prep.transform(X_test).astype('float32')
keras.utils.set_random_seed(42)
model = keras.Sequential([keras.layers.Input(shape=(A.shape[1],)),keras.layers.Dense(32,activation='relu'),keras.layers.Dropout(.2),keras.layers.Dense(16,activation='relu'),keras.layers.Dense(1,activation='sigmoid')])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=.001),loss='binary_crossentropy',metrics=['accuracy'])
model.fit(A,y_train.to_numpy(),validation_data=(B,y_val.to_numpy()),epochs=100,batch_size=32,verbose=1,callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',patience=12,restore_best_weights=True)])
val_prob = model.predict(B,verbose=0).ravel()
thresholds = np.arange(.2,.81,.01)
threshold = float(max(thresholds,key=lambda t:f1_score(y_val,val_prob>=t)))
prob = model.predict(C,verbose=0).ravel()
pred = (prob >= threshold).astype(int)
metrics = {'accuracy':accuracy_score(y_test,pred),'precision':precision_score(y_test,pred,zero_division=0),'recall':recall_score(y_test,pred,zero_division=0),'f1':f1_score(y_test,pred,zero_division=0),'roc_auc':roc_auc_score(y_test,prob),'threshold':threshold,'test_samples':len(y_test)}
model.save(MODEL_DIR/'churn_ann.keras')
joblib.dump(prep,MODEL_DIR/'preprocessor.joblib')
(MODEL_DIR/'metrics.json').write_text(json.dumps(metrics,indent=2))
print('Held-out test results:',json.dumps(metrics,indent=2))
