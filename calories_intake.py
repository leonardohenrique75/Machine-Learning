# -*- coding: utf-8 -*-

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree

#%%
# Leitura dos dados
caminho_arquivo = (
    r"C:\Projetos\Calories intake\fast_food_cheat_meal_workout_burn.csv"
)
database = pd.read_csv(caminho_arquivo)
#%%
# Cópia dos dados e remoção de colunas que não serão usadas
database_df = database.copy()
database_df = database_df.drop(
    columns=[
        "fitness_level",
        "fast_food_chain",
        "meal_category",
        "workout_shock_tier",
        "order_id",
    ],
    errors="ignore",
)
#%%
# Confira os dados
print(database_df.head())
print(database_df.dtypes)
#%%
# Transformação da variável categórica em colunas numéricas
database_df = pd.get_dummies(
    database_df,
    columns=["consumer_gender"],
    drop_first=True,
    dtype=int,
)
#%%
# Separação dos atributos e do alvo
# Não inclua um identificador artificial em X.
X = database_df.drop(columns=["sugar_grams"])
y = database_df["sugar_grams"]
#%%
# Divisão entre treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
)

#%%
def avaliar_modelo(nome, y_real, y_pred):
    """Exibe métricas de regressão."""
    mse = mean_squared_error(y_real, y_pred)
    mae = mean_absolute_error(y_real, y_pred)
    r2 = r2_score(y_real, y_pred)

    print(f"\n{nome}")
    print(f"MSE: {mse:.1f}")
    print(f"RMSE: {np.sqrt(mse):.1f}")
    print(f"MAE: {mae:.1f}")
    print(f"R²: {r2:.1%}")
#%%

# Árvore de decisão simples
tree_reg = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg.fit(X_train, y_train)

tree_pred_train = tree_reg.predict(X_train)
tree_pred_test = tree_reg.predict(X_test)

avaliar_modelo("Árvore simples — treino", y_train, tree_pred_train)
avaliar_modelo("Árvore simples — teste", y_test, tree_pred_test)
#%%
# Visualização da árvore simples
plt.figure(figsize=(20, 10))
plot_tree(
    tree_reg,
    feature_names=X.columns.tolist(),
    filled=True,
    node_ids=True,
    precision=2,
)
plt.tight_layout()
plt.show()
#%%
# Caminho de poda da árvore simples
path = tree_reg.cost_complexity_pruning_path(X_train, y_train)
pruning_table = pd.DataFrame({
    "ccp_alpha": path.ccp_alphas,
    "impurity": path.impurities,
})
print("\nOpções de poda:")
print(pruning_table.sort_values("ccp_alpha"))
#%%
# Busca de parâmetros para a árvore
param_grid_tree = {
    "max_depth": [3, 5, 10],
    "min_samples_split": [5, 10, 15],
    "min_samples_leaf": [5, 10],
}

tree_grid_search = GridSearchCV(
    estimator=DecisionTreeRegressor(random_state=42),
    param_grid=param_grid_tree,
    scoring="r2",
    cv=5,
    n_jobs=-1,
    verbose=2,
)
tree_grid_search.fit(X_train, y_train)
#%%
tree_best = tree_grid_search.best_estimator_
tree_grid_pred_train = tree_best.predict(X_train)
tree_grid_pred_test = tree_best.predict(X_test)

print("\nMelhores parâmetros da árvore:")
print(tree_grid_search.best_params_)

avaliar_modelo(
    "Árvore ajustada — treino",
    y_train,
    tree_grid_pred_train,
)
avaliar_modelo(
    "Árvore ajustada — teste",
    y_test,
    tree_grid_pred_test,
)
#%%
# Importância dos atributos da árvore ajustada
tree_features = pd.DataFrame({
    "feature": X.columns,
    "importance": tree_best.feature_importances_,
}).sort_values("importance", ascending=False)

print("\nImportância dos atributos — árvore ajustada:")
print(tree_features)
#%%
# Gráfico: valores reais versus previsões da árvore ajustada
tree_graph = pd.DataFrame({
    "sugar_grams": y_test,
    "predicted": tree_grid_pred_test,
})

plt.figure(figsize=(7, 6))
sns.scatterplot(data=tree_graph, x="sugar_grams", y="predicted", color="purple")
limite = max(y_test.max(), tree_grid_pred_test.max())
plt.plot([0, limite], [0, limite], color="grey", linestyle="--")
plt.title("Árvore ajustada: valores reais e previstos")
plt.xlabel("Açúcar real (g)")
plt.ylabel("Açúcar previsto (g)")
plt.tight_layout()
plt.show()
#%%
# Busca de parâmetros para o Random Forest
param_grid_rf = {
    "n_estimators": [100, 300],
    "max_features": [0.5, 0.8, 1.0],
    "max_depth": [None, 10, 20],
    "min_samples_leaf": [1, 2, 5],
}

rf_model = RandomForestRegressor(random_state=75)

time_start = time.time()

rf_grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid_rf,
    scoring="neg_root_mean_squared_error",
    cv=4,
    n_jobs=-1,
    verbose=2,
)
rf_grid_search.fit(X_train, y_train)

time_end = time.time()

best_model = rf_grid_search.best_estimator_

print("\nMelhores parâmetros do Random Forest:")
print(rf_grid_search.best_params_)
print(f"RMSE médio na validação cruzada: {-rf_grid_search.best_score_:.1f}")
print(f"Tempo da busca: {time_end - time_start:.1f} segundos")
#%%
# Avaliação do Random Forest
rf_pred_train = best_model.predict(X_train)
rf_pred_test = best_model.predict(X_test)

avaliar_modelo("Random Forest — treino", y_train, rf_pred_train)
avaliar_modelo("Random Forest — teste", y_test, rf_pred_test)

# Referência: previsão constante com a média do conjunto de treino
baseline_pred_test = np.full(len(y_test), y_train.mean())
avaliar_modelo("Referência da média — teste", y_test, baseline_pred_test)
#%%
# Importância dos atributos do Random Forest
rf_features = pd.DataFrame({
    "feature": X.columns,
    "importance": best_model.feature_importances_,
}).sort_values("importance", ascending=False)

print("\nImportância dos atributos — Random Forest:")
print(rf_features)
#%%
# Gráfico: valores reais versus previsões do Random Forest
rf_graph = pd.DataFrame({
    "sugar_grams": y_test,
    "predicted": rf_pred_test,
})

plt.figure(figsize=(7, 6))
sns.scatterplot(data=rf_graph, x="sugar_grams", y="predicted", color="teal")
limite = max(y_test.max(), rf_pred_test.max())
plt.plot([0, limite], [0, limite], color="grey", linestyle="--")
plt.title("Random Forest: valores reais e previstos")
plt.xlabel("Açúcar real (g)")
plt.ylabel("Açúcar previsto (g)")
plt.tight_layout()
plt.show()