import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
import joblib



# %%
# Assuming you have a CSV file named 'data.csv' in the current directory
df = pd.read_csv('data/data.csv')

# Display the dataframe
df.head()

# %%


# %%


# Assuming you have a DataFrame named 'df' with the required columns
# Assuming you have imported the necessary modules
# and have the dataframe 'df' defined

# Create an instance of the OneHotEncoder
one_hot_encoder = OneHotEncoder()

# Fit and transform the 'Brand' column
brand_encoded = one_hot_encoder.fit_transform(df[["Brand"]])
# Assuming you have a DataFrame named 'df' with the required columns
# Assuming you have imported the necessary modules
# and have the dataframe 'df' defined
# Insert the encoded column into the dataframe
df_encoded = pd.concat([df.drop("Brand", axis=1), pd.DataFrame(brand_encoded.toarray(), columns=one_hot_encoder.categories_[0], index=df.index)], axis=1)
df_encoded.rename(columns={"Brand_Encoded": "Brand"}, inplace=True)
df_encoded

df_encoded =df_encoded.drop(['Sort Code', 'Account Number', 'BIN'], axis=1)
df_encoded
#label_encoder = LabelEncoder()

columns_to_encode = ['Account Closed', 'Is this Acc linked to Sec in RMP','Is Security linked to any other active borrowing','Is all linked borrowings repaid','Type','Is Security good to release']
columns_to_rename = {'Account Closed': 'Account_Closed', 'Is this Acc linked to Sec in RMP': 'Sec_Linked', 'Is Security linked to any other active borrowing': 'Sec_Linked_to_Active_Borrowing', 'Is all linked borrowings repaid': 'All_Borrowings_Repaid', 'Type': 'Type', 'Is Security good to release': 'Outcome'}
columns_to_drop = ['Sec ID']  

label_encoder = LabelEncoder()

# Encode categorical columns and rename them
for column in columns_to_encode:
    df_encoded[column] = label_encoder.fit_transform(df_encoded[column])
    df_encoded.rename(columns={column: columns_to_rename.get(column, column)}, inplace=True)
df_encoded.drop(columns=columns_to_drop, inplace=True)
# Fit and transform the brand names
#df_encoded['Account Closed'] = label_encoder.fit_transform(df['Account Closed'])
#df_encoded['Account_Linked'] =  label_encoder.fit_transform(df['Is this Acc linked to Sec in RMP'])
#df_encoded.drop(columns=['Is this Acc linked to Sec in RMP'], inplace=True)
df_encoded


# %%
df_encoded.describe()

# %%
df_encoded['Outcome'].value_counts()

# %%
df_encoded['All_Borrowings_Repaid'].value_counts()

# %%
X = df_encoded.drop(columns = 'Outcome', axis=1)
Y = df_encoded['Outcome']

# %%
print(X)

# %%
print(Y)

# %%
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)

# %%
print(X.shape, X_train.shape, X_test.shape)

# %%
classifier = svm.SVC(kernel='linear')

# %%
classifier.fit(X_train, Y_train)

# %%
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

# %%
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

# %%
print('Accuracy score of the test data : ', test_data_accuracy)

# %%
input_data = (0,1,1,2,2,0.0,1.0)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = classifier.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('Security Cannot be Released')
else:
  print('Security Can be Released')

# %%
joblib.dump(classifier, 'classifier.pkl')



