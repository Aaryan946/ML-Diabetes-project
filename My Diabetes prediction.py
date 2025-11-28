import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,roc_auc_score,roc_curve
#Load data
df=pd.read_csv("c:\\Users\\Aaryan mitra\\OneDrive\\Desktop\\ML\\ml project\\diabetes.csv",encoding='latin1')
print(df.head())
print(df.info())
print(df.describe())
#Replace 0 with NaN in selected columns
cols_with_missing=['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
df[cols_with_missing]=df[cols_with_missing].replace(0,np.nan)
#Fill NaN with median(robust against outliers)
for col in cols_with_missing:
    df.fillna({col: df[col].median()}, inplace=True)
    print(df.isnull().sum())# check if missing values remai
X=df.drop('Outcome',axis=1)
y=df['Outcome']
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.2,random_state=42,stratify=y)
print("Training set size:",X_test.shape)
print("Test set size:",X_test.shape)
log_model=LogisticRegression(max_iter=1000)
log_model.fit(X_train,y_train)
y_pred_log=log_model.predict(X_test)
knn_model=KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train,y_train)
y_pred_knn=knn_model.predict(X_test)
rf_model=RandomForestClassifier(n_estimators=100,random_state=42)
rf_model.fit(X_train,y_train)
y_pred_rf=rf_model.predict(X_test)
print("Logistic Regression")
print("Accuracy:",accuracy_score(y_test,y_pred_log))


print("ROC-AUC:",roc_auc_score(y_test,log_model.predict_proba(X_test)[:,1]))
print(classification_report(y_test,y_pred_log))
print("Random Forest")
print("Accuracy:",accuracy_score(y_test,y_pred_rf))
print("ROC-AUC:",roc_auc_score(y_test,rf_model.predict_proba(X_test)[:,1]))
print(classification_report(y_test,y_pred_rf))
print("KNN")
print("Accuracy:",accuracy_score(y_test,y_pred_knn))
print("ROC-AUC:",roc_auc_score(y_test,knn_model.predict_proba(X_test)[:,1]))
print(classification_report(y_test,y_pred_knn))
cm=confusion_matrix(y_test,y_pred_rf)
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',
            xticklabels=["No Diabetes","Diabetes"],
            yticklabels=["No Diabetes","Diabetes"])
plt.xlabel("Predicted")
plt.ylabel('Actual')
plt.title("Confusion Matrix-RandomForest")
plt.show()
fpr,tpr,_=roc_curve(y_test,rf_model.predict_proba(X_test)[:,1])
plt.plot(fpr,tpr,label="RandomForest")
plt.plot([0,1],[0,1],'--',color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()





            
                         
            







    





