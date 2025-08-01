import pandas as pd
from sklearn.ensemble import RandomForestClassifer
from sklearn.model_selection import train_test_split
import joblib
import numpy as np

data = {
  'feature1': np.random.rand(100),
  'feature2': np.random.rand(100),
  'target': np.random.randint(0,1, 100)
       }
df = pd.DataFrame(data)

X_train, X_test, y_train, y_test = train_test_split(
  df[['feature1','feature2']], df['target], test_size=0.2)

model = RandomForestClassifer()
model.fit(X_train, Y_train)

print(f"Test accuracy: {model.score(X_test, y_test):.2f}")

#Save model
joblib.dump(model, 'model.pk1')
