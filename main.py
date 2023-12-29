import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np
import sklearn.model_selection


def print_separator(title=""):
    print("\n" + "=" * 30)
    if title:
        print(title)
    print("=" * 30)


def load_data(file_name):
    data = pd.read_csv(file_name, delimiter=';')
    pd.set_option('display.max_columns', 100)
    print_separator("Data Head")
    print(data.head())
    print_separator("All column names:")
    print(f' {data.columns}')
    return data.dropna(axis=0)


def select_features(data, feature_list):
    X = data[feature_list]
    y = data.alcohol
    return X, y


#
def train_validate_split(data):
    train_df, val_df = sklearn.model_selection.train_test_split(data,
                                                                test_size=0.2)  # The original dataset was split into two parts:
    train_X, train_y = train_df.drop(['quality'], axis=1), train_df.quality  # 80% for training and 20% for validation
    val_X, val_y = val_df.drop(['quality'], axis=1), val_df.quality
    print(train_X.shape, val_X.shape)


# The output (3918, 11) (980, 11) from the train_validate_split function shows the dimensions of your training and
# validation sets. train_X has 3918 samples with 11 features each, and val_X comprises 980 samples, also with 11
# features per sample.


def train_decision_tree(X, y, random_state=1):
    model = DecisionTreeRegressor(random_state=random_state)
    model.fit(X, y)
    return model


def evaluate_model(model, X, y, message="Predictions"):
    predictions = model.predict(X)
    print_separator(message)
    # for true, pred in zip(y, predictions):
    #     print(f"Actual: {true}, Predicted: {pred}")


def split_data(X, y, random_state=1):
    return train_test_split(X, y, random_state=random_state)


def calculate_mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def get_optimal_leaf_nodes(X_train, X_val, y_train, y_val, leaf_nodes_list):
    thelist = {}
    print_separator("Optimal Leaf Nodes Search")
    for max_leaf_nodes in leaf_nodes_list:
        model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
        model.fit(X_train, y_train)
        preds_val = model.predict(X_val)
        mae = calculate_mae(y_val, preds_val)
        thelist[max_leaf_nodes] = mae
        print(f"Max leaf nodes: {max_leaf_nodes} \t Mean Absolute Error: {mae}")
    optimal_nodes = min(thelist, key=thelist.get)
    return optimal_nodes, thelist


def train_random_forest( X_train, y_train, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=20,
                        max_features='auto', random_state=1):
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=None,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model


# # Main script
file = 'data/winequality-white.csv'
data = load_data(file)
train_validate_split(data)
features = ['total_sulfur_dioxide', 'pH', 'chlorides']
X, y = select_features(data, features)
train_X, val_X, train_y, val_y = split_data(X, y)

# Decision Tree with full data
dt_model_full = train_decision_tree(X, y)
evaluate_model(dt_model_full, X.head(10), y.head(10), "Initial Decision Tree Predictions")

# Decision Tree with train/validation split
dt_model_split = train_decision_tree(train_X, train_y)
evaluate_model(dt_model_split, val_X, val_y, "Decision Tree Validation Predictions")

# Find optimal leaf nodes
leaf_nodes_list = [5, 25, 50, 100, 250, 500, 750]
optimal_nodes, leaf_nodes_results = get_optimal_leaf_nodes(train_X, val_X, train_y, val_y, leaf_nodes_list)
print_separator(f"Optimal number of leaf nodes: {optimal_nodes}")

# Retrain with optimal leaf nodes
final_dt_model = DecisionTreeRegressor(max_leaf_nodes=optimal_nodes, random_state=1)
final_dt_model.fit(X, y)
evaluate_model(final_dt_model, X.head(20), y.head(20), "Optimal Leaf Nodes Predictions")

# Random Forest Model
rf_model = train_random_forest(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = calculate_mae(val_y, rf_val_predictions)
print_separator("Random Forest Model Evaluation")
print(f'Random Forest Model Mean Absolute Error: {rf_val_mae}')


# Function to create synthetic data
def create_synthetic_data(original_data, n_samples=10):
    # np.random.seed(1)

    # Randomly sample values from specified columns
    total_sulfur_dioxide = np.random.choice(original_data['total_sulfur_dioxide'], size=n_samples)
    pH = np.random.choice(original_data['pH'], size=n_samples)
    chlorides = np.random.choice(original_data['chlorides'], size=n_samples)

    synthetic_data = pd.DataFrame({
        'total_sulfur_dioxide': total_sulfur_dioxide,
        'pH': pH,
        'chlorides': chlorides
    })

    return synthetic_data


# Create synthetic data
new_data = create_synthetic_data(data, 10)

# Predict prices using the Random Forest model
new_predictions = rf_model.predict(new_data)

print("Synthetic Data:")
print(new_data)

# Print predictions
print("\nPredicted Average value:")
print(new_predictions)
