from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def train_model(feat_df):
    X = feat_df.drop(columns=["target"])
    y = feat_df["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    
    score = model.score(X_test, y_test)
    return model, score