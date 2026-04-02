def run_knn_model(df, dummy_column, drop_columns, n_neighbors=47):

    import pandas as pd
    import numpy as np
    import math
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.neighbors import KNeighborsRegressor

    # One-hot encoding
    df_encoded = pd.get_dummies(df, columns=dummy_column, dtype=int)

    # Features and target
    features = df_encoded.drop(columns=drop_columns)
    target = df_encoded["rating"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.20, random_state=0
    )

    # Scaling
    scaler = MinMaxScaler()
    scaler.fit(X_train)

    X_train_norm = scaler.transform(X_train)
    X_test_norm = scaler.transform(X_test)

    # Convert back to dataframe
    X_train_norm = pd.DataFrame(X_train_norm, columns=X_train.columns, index=X_train.index)
    X_test_norm = pd.DataFrame(X_test_norm, columns=X_test.columns, index=X_test.index)

    # Train KNN
    knn = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn.fit(X_train_norm, y_train)

    # Suggested K value
    k_suggested = int(math.sqrt(len(X_train)))

    # Model performance
    r2 = knn.score(X_test_norm, y_test)

    print("Suggested K:", k_suggested)
    print(f"R² on TEST set: {r2:.2f}")
    print(f"Accuracy: {r2*100:.2f}%")

    return knn, r2




#Statistical Inspection: K-S test
def ks_test(df):
    for col in df.columns:
        data = df[col].dropna()
        standardized = (data - data.mean()) / data.std()

        ks_test_statistic, ks_p_value = stats.kstest(standardized, 'norm')

        if ks_p_value < 0.05:
            print(f"{col}: distribution is significantly different from normal (p = {ks_p_value:})(t-statistic:{ks_test_statistic})")
        else:
            print(f"{col}: distribution is NOT significantly different from normal (p = {ks_p_value:})(t-statistic:{ks_test_statistic})")
