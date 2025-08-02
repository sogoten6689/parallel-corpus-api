import sys
import json
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

def main():
    try:
        # Đọc stdin (dữ liệu JSON)
        input_data = json.loads(sys.stdin.read())
        file_path = input_data['file_path']
        target_col = input_data.get('target_col', None)

        print(f"Processing file: {file_path}", file=sys.stderr)

        # Đọc file
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            # Try different engines for Excel files
            try:
                df = pd.read_excel(file_path, engine='openpyxl')
            except:
                try:
                    df = pd.read_excel(file_path, engine='xlrd')
                except:
                    df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
            
        print(f"File loaded successfully. Shape: {df.shape}", file=sys.stderr)

        # Nếu không có cột target, tự động chọn cột cuối cùng
        if not target_col or target_col not in df.columns:
            target_col = df.columns[-1]

        print(f"Using target column: {target_col}", file=sys.stderr)

        X = df.drop(columns=[target_col])
        y = df[target_col]

        print(f"Features shape: {X.shape}, Target shape: {y.shape}", file=sys.stderr)

        # Encode categorical features
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y.astype(str))

        # Train model
        model = DecisionTreeClassifier(max_depth=3, random_state=42)
        model.fit(X, y)

        # Feature importance
        importances = model.feature_importances_
        features = X.columns
        result = sorted(
            [{"feature": f, "importance": float(i)} for f, i in zip(features, importances)],
            key=lambda x: x["importance"], reverse=True
        )[:10]

        print(json.dumps({"feature_importance": result}))
        
    except Exception as e:
        print(f"XAI Error: {str(e)}", file=sys.stderr)
        print(json.dumps({"feature_importance": []}))

if __name__ == "__main__":
    main() 