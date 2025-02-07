import warnings
warnings.filterwarnings("ignore")
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.metrics import (precision_score, 
                             recall_score, 
                             f1_score, 
                             accuracy_score, 
                             matthews_corrcoef,
                             confusion_matrix)

from sklearn.linear_model import (RidgeClassifier, 
                                  LogisticRegression,
                                  SGDClassifier)

from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis)

from sklearn.svm import (LinearSVC, NuSVC, SVC)
from sklearn.neighbors import (KNeighborsClassifier, RadiusNeighborsClassifier)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import (BernoulliNB, CategoricalNB, GaussianNB)
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import (GradientBoostingClassifier, RandomForestClassifier, 
                              AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.model_selection import cross_validate
import numpy as np
import pandas as pd

class ClassificationModel:
    
    def __init__(self, 
                 train_values=None, 
                 test_values=None, 
                 train_response=None, 
                 test_response=None) -> None:
        
        self.train_values = train_values
        self.test_values = test_values
        self.train_response = train_response
        self.test_response = test_response
        
        self.scores = ['f1_weighted', 'recall_weighted', 'precision_weighted', 'accuracy']
        self.keys = ['fit_time', 'score_time', 'test_f1_weighted', 'test_recall_weighted', 'test_precision_weighted', 'test_accuracy']

    def __get_metrics(self, y_true=None, y_predict=None):
        # Obtener la matriz de confusión
        cm = confusion_matrix(y_true=y_true, y_pred=y_predict)
        
        # Inicializar las métricas
        sensitivity = []
        specificity = []
        
        # Calcular las métricas para cada clase
        for i in range(cm.shape[0]):
            tp = cm[i, i]  # Verdaderos positivos para la clase i
            fn = cm[i, :].sum() - tp  # Falsos negativos para la clase i
            fp = cm[:, i].sum() - tp  # Falsos positivos para la clase i
            tn = cm.sum() - (tp + fn + fp)  # Verdaderos negativos para la clase i

            sensitivity.append(tp / (tp + fn) if tp + fn != 0 else 0)
            specificity.append(tn / (tn + fp) if tn + fp != 0 else 0)

        # Calcular las métricas con promedio ponderado para multiclase
        accuracy = accuracy_score(y_true=y_true, y_pred=y_predict)
        recall = recall_score(y_true=y_true, y_pred=y_predict, average='weighted')
        precision = precision_score(y_true=y_true, y_pred=y_predict, average='weighted')
        f1 = f1_score(y_true=y_true, y_pred=y_predict, average='weighted')
        mcc = matthews_corrcoef(y_true=y_true, y_pred=y_predict)

        # Promediar sensibilidad y especificidad
        mean_sensitivity = np.mean(sensitivity)
        mean_specificity = np.mean(specificity)

        row = [
            accuracy,
            precision,
            recall,
            f1,
            mcc,
            mean_sensitivity,
            mean_specificity
        ]

        return row

    def __process_performance_cross_val(self, performances):
        row_response = []
        for i in range(len(self.keys)):
            value = np.mean(performances[self.keys[i]])
            row_response.append(value)
        return row_response
    
    def __apply_model(self, model=None, description=None):
        model.fit(self.train_values, self.train_response)
        predictions = model.predict(self.test_values)

        # Obtener métricas de validación
        metrics_validation = self.__get_metrics(
            y_true=self.test_response,
            y_predict=predictions)
        
        # Evaluación cruzada
        response_cv = cross_validate(
            model, 
            self.train_values, 
            self.train_response, 
            cv=5, 
            scoring=self.scores)

        metrics_cv = self.__process_performance_cross_val(
            response_cv
        )

        # Combinación de resultados
        row = [description] + metrics_cv + metrics_validation

        return row
    
    def apply_exploring(self):
        matrix_response = []

        try:
            clf_model = RidgeClassifier()
            matrix_response.append(
                self.__apply_model(
                    model=clf_model,
                    description="Ridge"
                )
            )
        except Exception as e:
            pass

        try:
            clf_model = LogisticRegression()
            matrix_response.append(
                self.__apply_model(
                    model=clf_model,
                    description="Logistic"
                )
            )
        except:
            pass
        
        try:
            clf_model = SGDClassifier(
                
            )
            matrix_response.append(
                self.__apply_model(
                    model=clf_model,
                    description="SGD"
                )
            )
        except:
            pass
        
        try:
            clf_model = LinearDiscriminantAnalysis()
            matrix_response.append(
                self.__apply_model(
                    model=clf_model,
                    description="LDA"
                )
            )
        except:
            pass
        
        try:
            clf_model = QuadraticDiscriminantAnalysis()
            matrix_response.append(
                self.__apply_model(
                    model=clf_model,
                    description="QDA"
                )
            )
        except:
            pass
        
        try:
            clf_model = LinearSVC(
                
            )
            matrix_response.append(
                self.__apply_model(
                    model=clf_model,
                    description="LinearSVC"
                )
            )
        except:
            pass
        
        try:
            clf_model = SVC(
                
            )
            matrix_response.append(
                self.__apply_model(
                    model=clf_model,
                    description="SVC"
                )
            )
        except:
            pass
        
        try:
            clf_model = NuSVC(
                
            )
            matrix_response.append(
                self.__apply_model(
                    model=clf_model,
                    description="NuSVC"
                )
            )
        except:
            pass
        
        try:
            clf_model = KNeighborsClassifier()
            matrix_response.append(
                self.__apply_model(
                    model=clf_model,
                    description="KNN"
                )
            )
        except:
            pass
        
        try:
            clf_model = RadiusNeighborsClassifier()
            matrix_response.append(
                self.__apply_model(
                    model=clf_model,
                    description="RNN"
                )
            )
        except:
            pass
        
        try:
            clf_model = GaussianProcessClassifier()
            matrix_response.append(
                self.__apply_model(
                    model=clf_model,
                    description="GPC"
                )
            )
        except:
            pass
        
        try:
            clf_model = BernoulliNB()
            matrix_response.append(
                self.__apply_model(
                    model=clf_model,
                    description="BernoulliNB"
                )
            )
        except:
            pass
        
        try:
            clf_model = CategoricalNB()
            matrix_response.append(
                self.__apply_model(
                    model=clf_model,
                    description="CategoricalNB"
                )
            )
        except:
            pass
        
        try:
            clf_model = GaussianNB()
            matrix_response.append(
                self.__apply_model(
                    model=clf_model,
                    description="GaussianNB"
                )
            )
        except:
            pass
        
        try:
            clf_model = DecisionTreeClassifier(
                
            )
            matrix_response.append(
                self.__apply_model(
                    model=clf_model,
                    description="DecisionTree"
                )
            )
        except:
            pass
        
        try:
            clf_model = ExtraTreeClassifier(
                
            )
            matrix_response.append(
                self.__apply_model(
                    model=clf_model,
                    description="ExtraTree"
                )
            )
        except:
            pass
        
        try:
            clf_model = RandomForestClassifier(
                
            )
            matrix_response.append(
                self.__apply_model(
                    model=clf_model,
                    description="RandomForest"
                )
            )
        except:
            pass

        try:
            clf_model = BaggingClassifier()
            matrix_response.append(
                self.__apply_model(
                    model=clf_model,
                    description="Bagging"
                )
            )
        except:
            pass
        
        try:
            clf_model = AdaBoostClassifier()
            matrix_response.append(
                self.__apply_model(
                    model=clf_model,
                    description="AdaBoost"
                )
            )
        except:
            pass
        
        try:
            clf_model = GradientBoostingClassifier()
            matrix_response.append(
                self.__apply_model(
                    model=clf_model,
                    description="GradientBoosting"
                )
            )
        except:
            pass
        
        try:
            clf_model = HistGradientBoostingClassifier(
                
            )
            matrix_response.append(
                self.__apply_model(
                    model=clf_model,
                    description="HistGradientBoosting"
                )
            )
        except:
            pass
        
        try:
            clf_model = ExtraTreesClassifier(
                
            )
            matrix_response.append(
                self.__apply_model(
                    model=clf_model,
                    description="ExtraTrees-ensemble"
                )
            )
        except:
            pass
        
        try:
            clf_model = XGBClassifier()
            matrix_response.append(
                self.__apply_model(
                    model=clf_model,
                    description="XGB"
                )
            )
        except:
            pass
        
        try:
            clf_model = LGBMClassifier()
            matrix_response.append(
                self.__apply_model(
                    model=clf_model,
                    description="LGBM"
                )
            )
        except:
            pass

        header = ["algorithm", 'fit_time', 'score_time', 'F1_cv', 'recall_cv', 'precision_cv', 'accuracy_cv', 
                  'accuracy_val', 'precision_val', 'recall_val', 'f1_val', 'matthews_corrcoef_val', 'sensitivity', 'specificity']
        
        df_summary = pd.DataFrame(data=matrix_response, 
                                  columns=header)

        return df_summary