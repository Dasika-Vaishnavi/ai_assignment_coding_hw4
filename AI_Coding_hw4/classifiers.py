from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from matplotlib import colormaps
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class Classifiers():
    def __init__(self, data):
        # Features are the first two columns (A, B); label is the last column.
        X = data[['A', 'B']].values
        y = data['label'].values

        # 60/40 train/test split, stratified so class proportions are preserved.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, stratify=y, random_state=42
        )

        self.training_data = X_train
        self.training_labels = y_train
        self.testing_data = X_test
        self.testing_labels = y_test
        self.outputs = []

    def test_clf(self, clf, classifier_name=''):
        """Fit the GridSearchCV object, record best CV score and test score, then plot."""
        clf.fit(self.training_data, self.training_labels)

        best_train_score = clf.best_score_                                    # mean 5-fold CV accuracy
        test_score = clf.score(self.testing_data, self.testing_labels)        # accuracy on held-out test set

        print(f'  Best params:    {clf.best_params_}')
        print(f'  Best CV score:  {best_train_score:.4f}')
        print(f'  Test score:     {test_score:.4f}')

        self.outputs.append(f'{classifier_name},{best_train_score},{test_score}')

        # Plot decision boundary using the test set
        self.plot(self.testing_data, self.testing_labels,
                  model=clf, classifier_name=classifier_name)

    def classifyNearestNeighbors(self):
        param_grid = {
            'n_neighbors': list(range(1, 20, 2)),     # [1, 3, 5, ..., 19]
            'leaf_size':   list(range(5, 31, 5)),     # [5, 10, 15, 20, 25, 30]
        }
        clf = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
        self.test_clf(clf, classifier_name='k-Nearest Neighbors')

    def classifyLogisticRegression(self):
        param_grid = {
            'C': [0.1, 0.5, 1, 5, 10, 50, 100],
        }
        clf = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5)
        self.test_clf(clf, classifier_name='Logistic Regression')

    def classifyDecisionTree(self):
        param_grid = {
            'max_depth':         list(range(1, 51)),  # 1..50
            'min_samples_split': list(range(2, 11)),  # 2..10
        }
        clf = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5)
        self.test_clf(clf, classifier_name='Decision Tree')

    def classifyRandomForest(self):
        param_grid = {
            'max_depth':         [1, 2, 3, 4, 5],
            'min_samples_split': list(range(2, 11)),
        }
        clf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
        self.test_clf(clf, classifier_name='Random Forest')

    def classifyAdaBoost(self):
        param_grid = {
            'n_estimators': list(range(10, 71, 10)),  # [10, 20, ..., 70]
        }
        clf = GridSearchCV(AdaBoostClassifier(random_state=42), param_grid, cv=5)
        self.test_clf(clf, classifier_name='AdaBoost')

    def plot(self, X, Y, model, classifier_name=''):
        X1 = X[:, 0]
        X2 = X[:, 1]

        X1_min, X1_max = min(X1) - 0.5, max(X1) + 0.5
        X2_min, X2_max = min(X2) - 0.5, max(X2) + 0.5

        X1_inc = (X1_max - X1_min) / 200.
        X2_inc = (X2_max - X2_min) / 200.

        X1_surf = np.arange(X1_min, X1_max, X1_inc)
        X2_surf = np.arange(X2_min, X2_max, X2_inc)
        X1_surf, X2_surf = np.meshgrid(X1_surf, X2_surf)

        L_surf = model.predict(np.c_[X1_surf.ravel(), X2_surf.ravel()])
        L_surf = L_surf.reshape(X1_surf.shape)

        plt.figure(figsize=(6, 5))
        plt.title(classifier_name)
        plt.contourf(X1_surf, X2_surf, L_surf, cmap=plt.cm.coolwarm, zorder=1)
        plt.scatter(X1, X2, s=38, c=Y, edgecolors='k')
        plt.xlabel('A')
        plt.ylabel('B')
        plt.margins(0.0)
        plt.savefig(f'{classifier_name}.png', dpi=120, bbox_inches='tight')
        plt.close()


def plot_dataset(df):
    """Part (a): scatter plot showing the two classes with two different patterns."""
    plt.figure(figsize=(6, 5))
    cls0 = df[df['label'] == 0]
    cls1 = df[df['label'] == 1]
    plt.scatter(cls0['A'], cls0['B'], marker='o', facecolors='none',
                edgecolors='red',  s=45, label='Class 0')
    plt.scatter(cls1['A'], cls1['B'], marker='x',
                color='blue', s=45, label='Class 1')
    plt.xlabel('A')
    plt.ylabel('B')
    plt.title('Input Dataset')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('dataset_scatter.png', dpi=120, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    df = pd.read_csv('input.csv')

    # Part (a): visualize the raw dataset
    plot_dataset(df)

    models = Classifiers(df)
    print('Classifying with NN...')
    models.classifyNearestNeighbors()
    print('Classifying with Logistic Regression...')
    models.classifyLogisticRegression()
    print('Classifying with Decision Tree...')
    models.classifyDecisionTree()
    print('Classifying with Random Forest...')
    models.classifyRandomForest()
    print('Classifying with AdaBoost...')
    models.classifyAdaBoost()

    with open("output.csv", "w") as f:
        print('Name, Best Training Score, Testing Score', file=f)
        for line in models.outputs:
            print(line, file=f)
