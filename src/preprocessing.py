from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd

def create_aktywny_klient(df):
    df['aktywny_klient'] = df['dni_od_ostatniego_zakupu'].apply(lambda x: 1 if x <= 90 else 0)
    return df

def encode_categorical(df):
    le = LabelEncoder()
    df['plec_encoded'] = le.fit_transform(df['plec'])
    df = pd.get_dummies(df, columns=['miasto'], drop_first=True)
    df = df.drop(columns=['klient_id', 'plec'])
    return df

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def scale_all_data(x):
    scaler = StandardScaler()
    return scaler.fit_transform(x)