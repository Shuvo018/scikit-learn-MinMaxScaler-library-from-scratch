class MinMaxScaler:
    # Here get min and max value
    def fit(self, X_train):
        self.classes_name = []
        self.minvalues = []
        self.maxvalues = []
        for x in X_train:
            self.classes_name.append(x)
            self.minvalues.append(min(X_train[x]))
            self.maxvalues.append(max(X_train[x]))

        # print(self.minvalues)
        # print(self.maxvalues)

# changing original value
    def transform(self, X_values):
        self.X_values = X_values
        cnt = 0
        for i,cn in enumerate(self.classes_name):
            # down = (self.maxvalues[i] - self.minvalues[i])
            for j, x in enumerate(X_values[cn]):

                # v = (x-self.minvalues[i])/(down)
                v = (x-self.minvalues[i])/(self.maxvalues[i] - self.minvalues[i])
                self.X_values[cn][j] = v

                # if cnt<5:
                #     cnt += 1
                #     print(f'{x} - {self.minvalues[i]}  / {self.maxvalues[i]} - {self.minvalues[i]}')
                    
        return self.X_values

# import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv('wine_data.csv',header=None,usecols=[0,1,2])
df.columns=['Class label', 'Alcohol', 'Malic acid']

# spliting data
X = df.drop('Class label', axis=1)
y = df['Class label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = MinMaxScaler()

# scaler.fit(X)
# X_scaler = scaler.transform(X)

scaler.fit(X_train)

X_train_scaler = scaler.transform(X)
X_test_scaler = scaler.transform(X_test)

print(X_train_scaler.head())
# print(np.round(X_train_scaler.describe(), 1))
