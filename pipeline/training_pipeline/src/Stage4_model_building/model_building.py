import datetime,sys
import contextlib
from scipy.sparse import load_npz
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics  import roc_auc_score
from utils.utils import FileOperation
from app_tracking.logger import App_Logger
from app_tracking.exception import AppException

class ModelTraining:
    def __init__(self, current_time):
        try:
            self.current_time = current_time
            self.filepath = f"artifacts/logs/Stage4_ModelTraining/{self.current_time}.txt"
            self.logging = App_Logger(self.filepath)
            self.fileoperation = FileOperation()
            self.X_train = load_npz('artifacts/data/X_trainSparse_matrix.npz').toarray()
            self.X_test = load_npz('artifacts/data/X_testSparse_matrix.npz').toarray()
            self.y_train_path = "artifacts/data/y_train.csv"
            self.y_train = self.fileoperation.load_data_from_csv(self.y_train_path)
            self.y_test_path = "artifacts/data/y_test.csv"
            self.y_test = self.fileoperation.load_data_from_csv(self.y_test_path)
            self.finalTrain = load_npz('artifacts/data/Final_trainSparse_matrix.npz').toarray()
            self.target_path = "artifacts/data/FinalTargetData.csv" 
            self.target = self.fileoperation.load_data_from_csv(self.target_path)
            self.fileoperation.delete_files_in_directory("artifacts/models/TrainAndTest")

            with contextlib.suppress():
                self.create_and_train_model()
                self.bestmodel()

            with contextlib.suppress():
                self.bestModelForDeploy()
        except Exception as e:
            self.logging.log(str(e))
            raise AppException(e,sys) from e
        
    def create_and_train_model(self):
        try:
            print("Model Training has been Started")
            self.logging.log("Model Training has been Started")

            self.logging.log("creating and fitting GradientBoostingClassifier model on cluster 1")
            self.model1 = GradientBoostingClassifier(learning_rate = 0.2, 
                                                    n_estimators = 200).fit(self.X_train,self.y_train)
            self.logging.log(f"GradientBoostingClassifier model 1 on clustor 1 {self.model1} has been trained")  
            self.fileoperation.save_model(self.model, "artifacts/models/TrainAndTest/GradientBoostingClassifier1.pkl")
            self.logging.log(f"GradientBoostingClassifier model has been saved to artifacts/models/TrainAndTest/GradientBoostingClassifier1.pkl") 


            self.logging.log("creating and fitting GradientBoostingClassifier model 2 on cluster 2")
            self.model2 = GradientBoostingClassifier(learning_rate = 0.2, 
                                                    n_estimators = 200).fit(self.X_train,self.y_train)
            self.logging.log(f"GradientBoostingClassifier model 2 on clustor 2 {self.model1} has been trained")  
            self.fileoperation.save_model(self.model, "artifacts/models/TrainAndTest/GradientBoostingClassifier2.pkl")
            self.logging.log(f"GradientBoostingClassifier model has been saved to artifacts/models/TrainAndTest/GradientBoostingClassifier2.pkl")    


            self.logging.log("creating and fitting GradientBoostingClassifier model 3 on cluster 3")
            self.model2 = GradientBoostingClassifier(learning_rate = 0.2, 
                                                    n_estimators = 200).fit(self.X_train,self.y_train)
            self.logging.log(f"GradientBoostingClassifier model 3 on clustor 3 {self.model1} has been trained")  
            self.fileoperation.save_model(self.model, "artifacts/models/TrainAndTest/GradientBoostingClassifier3.pkl")
            self.logging.log(f"GradientBoostingClassifier model has been saved to artifacts/models/TrainAndTest/GradientBoostingClassifier3.pkl")    
            
            self.logging.log("Models Training have been Done Successfully")
        except Exception as e:
            self.logging.log(str(e))
            raise AppException(e,sys) from e
        

    






        





