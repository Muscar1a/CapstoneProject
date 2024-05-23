import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

def pre_process(df):
    if (df['gender'] == 'female' or df['gender'] == 'Female'):
        df['gender'] = 0
    else:
        df['gender'] = 1
        
    if (df['ever_married'] == 'yes' or df['ever_married'] == 'Yes'):
        df['ever_married'] = 1
    else:
        df['ever_married'] = 0
        
    # if (df['work_type'] == 'Private' or df['work_type'] == 'private'):
        # df['gender'] = 2
    if (df['work_type'] == 'Self-employed' or df['work_type'] == 'self-employed'):
        df['work_type'] = 3
    elif (df['work_type'] == 'Govt_job' or df['work_type'] == 'govt_job'):
        df['work_type'] = 0
    elif (df['work_type'] == 'children'):
        df['work_type'] = 4
    elif (df['work_type'] == 'Never_worked' or df['work_type'] == 'never_worked'):
        df['work_type'] = 1
    else:
        df['work_type'] = 2
        
    if (df['Residence_type'] == 'Urban' or df['Residence_type'] == 'urban'):
        df['Residence_type'] = 1
    else:
        df['Residence_type'] = 0
        
    if (df['smoking_status'] == 'smokes'):
        df['smoking_status'] = 3
    elif (df['smoking_status'] == 'formerly smoked'):
        df['smoking_status'] = 1
    elif (df['smoking_status'] == 'never smoked'):
        df['smoking_status'] = 2
    else:
        df['smoking_status'] = 0
    return df

def get_data():
    input_data = {}
    print("Enter your gender (Male/Female): ")
    input_data['gender'] = input()
    print("Enter your age: ")
    input_data['age'] = input()
    print('Do you have hypertension? ("1" for yes and "0" for no): ')
    input_data['hypertension'] = input()
    print('Do you have heart disease? ("1" for yes and "0" for no): ')
    input_data['heart_disease'] = input()
    print('Have you ever been married? (Yes/No): ')
    input_data['ever_married'] = input()
    print("Enter your work type (children, Govt_job, Never_worked, Private or Self-employed): ")
    input_data['work_type'] = input()
    print("Enter your residence type (Urban/Rural): ")
    input_data['Residence_type'] = input()
    print("Enter your average glucose level: ")
    input_data['avg_glucose_level'] = input()
    print("Enter your BMI: ")
    input_data['bmi'] = input()
    print("Enter your smoking status (smokes/formerly smoked/never smoked/Unknown): ")
    input_data['smoking_status'] = input()
    return input_data
    
def pre_process_df(df):
    df = df.drop('id', axis=1)
    le = LabelEncoder()
    df['gender'] = le.fit_transform(df['gender'])
    df['ever_married'] = le.fit_transform(df['ever_married'])
    df['work_type'] = le.fit_transform(df['work_type'])
    df['Residence_type'] = le.fit_transform(df['Residence_type'])
    df['smoking_status'] = le.fit_transform(df['smoking_status'])
    return df

def insert_file_test(model):
    df = pd.read_csv("D:\\Learn\\Uni\\ML\\CapstoneProject\\data\\test-data.csv")
    df = pre_process_df(df)
    y = df['stroke']
    X = df.drop(columns=['stroke'])
    svm_predict = model.predict(X)
    print(accuracy_score(y, svm_predict))
    print(svm_predict)
    
def main(): 
    svm = joblib.load("D:\\Learn\\Uni\\ML\\CapstoneProject\\src\\model\\svmrbf.pkg")
    # rf = joblib.load("D:\\Learn\\Uni\\ML\\CapstoneProject\\src\\model\\rf.pkg")
    # xgb = joblib.load("D:\\Learn\\Uni\\ML\\CapstoneProject\\src\\model\\xgboost.pkg")
    

    
    # insert_file_test(svm)
    
    data = get_data()
    
    data = pre_process(data)
    X = pd.DataFrame([data])
    svm_predict = svm.predict(X)
    print("---------------------")
    print("The result is: ", svm_predict[0], sep='')
    
    
    # print(accuracy_score(y, svm_predict))
    
    # rf_predict = rf.predict(X)
    # print(accuracy_score(y, rf_predict))
    
    # xgb_predict = xgb.predict(X)
    # print(accuracy_score(y, xgb_predict))
    
    
if __name__ == "__main__":
    main()