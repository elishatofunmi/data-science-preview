import numpy as np
import pandas as pd

class apply_module:
    def __init__(self):
        self.none_int_float = []
        return
    
    
    def check_if_categorical(data):
        """
        #Your input to the function is an array, 
        #it returns either true or false using the pandas
        #library .dtype.name method
        """
        solution = pd.Categorical(data)
        categorical = solution.dtype.name
        return categorical == 'category'
    
    def get_numerical_columns(data):
        """
        returns numerical data in your pandas dataframe.
        your input is a pandas dataframe.
        """
        dataframe_columns = data.columns
        numerical_columns = data.get_numeric_data().columns
        return numerical_columns
    
    def check_if_float_or_int(data):
        """
        Checks if a particular data is of type float or type int
        return true or false
        """
        none_int_float = []
        if data.dtypes.name == 'int64' or data.dtypes.name == 'float64':
            test = True
        else:
            test = False
        return test
        
    def _is_cat_ordinal(data):
        # checks if the data entry is categorical or ordinal
        
        dict_data = data.value_counts()
        if len(dict_data.keys()) > 15:
            value = False
            shape = len(dict_data.keys())
        else:
            value = True
            shape = len(dict_data.keys())
        return value, shape
    
    
    
class preprocessor:
    def __init__(self):
        """
        Helps to scale down, normalize the data
        """
        return
    
    def apply_scale(data):
        from sklearn.preprocessing import StandardScaler
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.preprocessing import Normalizer
        
        std_scaler = StandardScaler()
        max_scaler = MinMaxScaler()
        norm = Normalizer()
        
        x_train_scale = std_scaler.fit_transform(data)
        x_train_max = max_scaler.fit_transform(x_train_scale)
        x_train_norm = norm.fit_transform(x_train_max)
        return x_train_norm
    
    
    def apply_validation(x, y, train_size = 0.7, random_state = 32):
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = train_size, random_state = random_state)
        return x_train, x_test, y_train, y_test
    
    

    
    
class build_my_model:
    def __init__(self):
        self.model_type = None
        return
    
    def set_model_type(model):
        self.model_type = model
        return
    
    def apply_model(model):
        return
    
    
    
class apply_to_test_data:
    def __init__(self):
        return
    
    def apply(data):
        
        
    
    