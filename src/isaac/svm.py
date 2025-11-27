import marimo

__generated_with = "0.16.5"
app = marimo.App(width="columns")

with app.setup:
    # Initialization code that runs before all other cells
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.metrics import roc_curve,auc,confusion_matrix,precision_score,recall_score, f1_score
    from preprocessing import preprocess,get_labels
    from extraction import feature_extraction
    import pandas as pd
    import numpy as np
    import marimo as mo
    import seaborn as sns
    import matplotlib.pyplot as plt


@app.cell(hide_code=True)
def _():
    mo.md("""# Load and Extract Features from Dataset""")
    return


@app.cell
def _():
    df = pd.read_csv("data/train_data.csv",index_col=0)
    X = df.pipe(preprocess,attr= [])
    X
    return X, df


@app.cell
def _(df):
    y = get_labels(df)
    y
    return (y,)


@app.cell(hide_code=True)
def _():
    mo.md("""# Create Model""")
    return


@app.cell
def _():
    svm = SVC(probability=True)
    svm
    return (svm,)


@app.cell(column=1, hide_code=True)
def _():
    mo.md("""# Hyper-Parameter Optimization""")
    return


@app.cell
def _():
    grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale','auto',0.01,0.1],
        'kernel': ['rbf','poly','sigmoid']
    }
    return (grid,)


@app.cell
def _(grid, svm):
    search = GridSearchCV(
        estimator=svm,
        param_grid=grid,
        cv = 5,
        scoring='accuracy',
        n_jobs=3
    )
    return (search,)


@app.cell
def _(X, search, y):
    search.fit(X=X, y=y.values.flatten())
    return


@app.cell
def _(search):
    mo.md(f"""*Best training score for SVM is {search.best_score_:.2f}*""")
    return


@app.cell
def _():
    test = pd.read_csv("data/test_data.csv",index_col=0)
    X_test = test.pipe(preprocess, attr = [])
    y_test = test.pipe(get_labels)

    return X_test, y_test


@app.cell(column=2)
def _(X_test, search, y_test):
    optimal = search.best_estimator_
    y_score = optimal.predict_proba(X_test)
    y_pred = y_score.argmax(axis = 1)
    cm = confusion_matrix(
        y_test['Plant_Health_Status'].astype('category').cat.codes,
        y_pred
    )
    return cm, y_pred, y_score


@app.cell(hide_code=True)
def _():
    mo.md(
        f"""
    # SVM Performance
    SVM struggled to differentiate the level of stress of plants rather than the binary case of whether a plant was stressed or not
    """
    )
    return


@app.cell(hide_code=True)
def _(cm, y):
    sns.heatmap(
        cm,annot=True,cmap='mako',
        xticklabels=y["Plant_Health_Status"].unique(),
        yticklabels=y["Plant_Health_Status"].unique()
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    return


@app.cell
def _():
    return


@app.cell
def _(y_test):
    o = OneHotEncoder(sparse_output=False)
    y_classes = o.fit_transform(y_test)

    return o, y_classes


@app.cell(column=3, hide_code=True)
def _():
    mo.md(
        f"""
    # ROC Curves Per-Class
    Both the Healthy and High Stress Classes Have been predicted well while the 
    Moderate Stress class has significantly lower performance overall
    """
    )
    return


@app.cell(hide_code=True)
def _(o, y_classes, y_score):
    tprs = []
    fprs = []
    names = [f.split("_")[-1] for f in o.get_feature_names_out()]
    plt.figure(figsize=(8,6))
    for i in range(y_score.shape[1]):
        tpr,fpr, _ =roc_curve(y_classes[:,i],y_score[:,i])
        a = auc(tpr,fpr)
        tprs.append(tpr)
        fprs.append(fpr)
        plt.plot(tpr,fpr,label = f"{names[i]} (AUC={a:.2f})")
    plt.legend(bbox_to_anchor = (1.08,1.1), ncol = 3,frameon = False)
    plt.grid(True)
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(f"""# Recall Precision and F1""")
    return


@app.cell
def _(y_pred, y_test):
    precision = precision_score(y_pred,y_test['Plant_Health_Status'].astype('category').cat.codes,average='micro')
    recall = recall_score(y_pred,y_test['Plant_Health_Status'].astype('category').cat.codes,average='micro')
    f1 = f1_score(y_pred,y_test['Plant_Health_Status'].astype('category').cat.codes,average='micro')
    return f1, precision, recall


@app.cell
def _(f1, precision, recall):
    mo.md(
        f"""
    Average Precision: {precision:.2f}\n
    Average Recall: {recall:.2f}\n
    Average F1: {f1:.2f}
    """
    )
    return


@app.function(hide_code=True)
def svm_train(X,y):
    """
        Trains SVM model on dataset
        returns the predicted labels, the probability of each class
        and the optimal model
    """
    svm = SVC(probability=True)
    grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale','auto',0.01,0.1],
        'kernel': ['rbf','poly','sigmoid']
    }
    search = GridSearchCV(
        estimator=svm,
        param_grid=grid,
        cv = 5,
        scoring='accuracy',
        n_jobs=1
    )
    search.fit(X,y.values.flatten())
    optimal = search.best_estimator_
    y_score = optimal.predict_proba(X)
    y_pred = y_score.argmax(axis = 1)
    return y_pred, y_score, optimal


if __name__ == "__main__":
    app.run()
