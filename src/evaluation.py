from sklearn.model_selection import cross_val_score
import pandas as pd

def evaluate_with_cv(models, X, y, scoring='f1', cv=5):
    results = []
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        results.append({
            'Model': name,
            f'Mean {scoring.upper()} (CV)': scores.mean(),
            'Std Dev': scores.std()
        })
    return pd.DataFrame(results).sort_values(by=f'Mean {scoring.upper()} (CV)', ascending=False)