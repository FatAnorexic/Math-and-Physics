from sklearn.svm import SVC as svc
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import MinMaxScaler
scale=MinMaxScaler()
can=load_breast_cancer()

xt, xtest, yt, ytest=tts(can.data, can.target, random_state=1)

svm=svc(C=100)
"""
svm.fit(xt,yt)

print(f'Test set accuracy: {svm.score(xtest,ytest):.2f}')

#This generated an accuracy of about 94%, but now we'll use
#minmax scaler before fitting svc
"""

xtrain_scaled=scale.fit_transform(xt)
xtest_scaled=scale.transform(xtest)
svm.fit(xtrain_scaled, yt)

print(f'Scaled test set: {svm.score(xtest_scaled,ytest):.2f}')

#This yields an accuracy of about 97%