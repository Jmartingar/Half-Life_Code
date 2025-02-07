import pandas as pd
import sys
import math
from scipy.fft import fft
import numpy as np

class FFTTransform:

    def __init__(
            self,
            dataset=None,
            size_data=None,
            columns_to_ignore=[]):
        
        self.size_data = size_data
        self.dataset = dataset
        self.columns_to_ignore = columns_to_ignore

        self.init_process()

    def __processing_data_to_fft(self):

        print("Removing columns data")
        
        if len(self.columns_to_ignore) >0:
            self.data_ignored = self.dataset[self.columns_to_ignore]
            self.dataset = self.dataset.drop(columns=self.columns_to_ignore)

        self.dataset = self.dataset.select_dtypes(include=[np.number])

    # Determina el valor más cercano a una potencia de 2, ya que FFT requiere que el tamaño de la señal sea una potencia de 2
    def __get_near_pow(self):

        print("Get near pow 2 value")
        list_data = [math.pow(2, i) for i in range(1, 20)]
        stop_value = list_data[0]

        for value in list_data:
            if value >= self.size_data:
                stop_value = value
                break

        self.stop_value = int(stop_value)
    
    def __complete_zero_padding(self):

        print("Apply zero padding")
        list_df = [self.dataset]
        for i in range(self.size_data, self.stop_value):
            column = [0 for k in range(len(self.dataset))]
            key_name = "p_{}".format(i)
            df_tmp = pd.DataFrame()
            df_tmp[key_name] = column
            list_df.append(df_tmp)

        self.dataset = pd.concat(list_df, axis=1)
    

    def init_process(self):
        self.__processing_data_to_fft()
        self.__get_near_pow()
        self.__complete_zero_padding()

    def __create_row(self, index):
        row =  self.dataset.iloc[index].tolist()
        return row
    
    def __apply_FFT(self, index):

        row = self.__create_row(index)
        T = 1.0 / float(self.stop_value)
        yf = fft(row)

        xf = np.linspace(0.0, 1.0 / (2.0 * T), self.stop_value // 2)
        yf = np.abs(yf[0:self.stop_value // 2])
        return [value for value in yf]


    def encoding_dataset(self):

        matrix_response = []
        for index in self.dataset.index:
            row_fft = self.__apply_FFT(index)
            matrix_response.append(row_fft)

        print("Creating dataset")
        header = ['p_{}'.format(i) for i in range(len(matrix_response[0]))]
        print("Export dataset")
        df_fft = pd.DataFrame(matrix_response, columns=header)
        
        if len(self.columns_to_ignore)>0:

            df_fft = pd.concat([df_fft, self.data_ignored], axis=1)

        return df_fft

df_coded_processed = pd.read_csv(sys.argv[1])
name_output = sys.argv[2]
name_column = sys.argv[3]
column_with_response = sys.argv[4]

fft_instance = FFTTransform(
    dataset=df_coded_processed,
    size_data=len(df_coded_processed.columns)-1,
    columns_to_ignore=[column_with_response]
)

response_fft = fft_instance.encoding_dataset()
response_fft.to_csv(name_output, index=False)
print(response_fft.head(5))
