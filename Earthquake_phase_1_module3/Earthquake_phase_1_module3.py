import tensorflow as tf
import numpy as np
from tensorflow import keras
import pandas as pd
import datetime as dt

class EarthQuakePhase1:
    
    def __init__(self, *args, **kwargs):
        self.MIN_RANGE=111965
        self.MAX_RANGE=12312022
        self.datasetList=[]
        self.testData=[]
        self.dataset1=''
        self.dataset2=''
        self.dataset=''
        self.next=''
        self.date=''
        self.magnitude=''
        self.CSV_PATH='./database.csv'
        self.BATCH_SIZE=50
        self.expectedOutput=[]
        h_prev=0
        self.colNames=['Date','Magnitude']
        self.colNames_Graphs=['Latitude','Longitude','Magnitude']
        self.colNames_LSTM=['Date','Latitude','Longitude']
        self.temp_data=[]
        self.temp_data_year=pd.DataFrame(columns=('Year','Magnitude'))
        self.next_state=''
        self.init_state=''
        #80 years considered for global max
        self.global_date_max=float((pd.to_datetime('1/1/1970')+pd.Timedelta(days=365*80)).to_datetime64())
        return super().__init__(*args, **kwargs)

    def loadData_Graph(self):
        self.temp_data=pd.read_csv(self.CSV_PATH,usecols=self.colNames_Graphs)
        Latitude=self.temp_data['Latitude'].values
        Longitude=self.temp_data['Longitude'].values
        Magnitude=self.temp_data['Magnitude'].values
        print('in load')

        return Latitude,Longitude,Magnitude

    def loadData_LSTM(self):
        self.temp_data=pd.read_csv(self.CSV_PATH,usecols=self.colNames_LSTM)
        Date=self.temp_data['Date'].values
        
        #Latitude=self.temp_data['Latitude'].values
        #Longitude=self.temp_data['Longitude'].values
        #Magnitude=self.temp_data['Magnitude'].values
        
        Date=self.convertDateToMS(Date)
        print('in load',type(Date),type(Date[0]),Date[0],len(Date),self.global_date_max)
        trainData=Date[:20001]
        self.testData=Date[20001:]
        #Date=Date[:51]
        #self.testData=Date[51:100]
        #self.testData=np.array(self.testData)*0.001         #factor 0.001 to reduce the value
        trainData.sort()
        self.testData.sort()
        #Date=np.array(Date)*0.001
        print('in load',type(Date),type(Date[0]),Date[0],len(Date))
        
        #self.mergeData(Date)

        return trainData,self.global_date_max,self.testData #,self.expectedOutput

    def mergeData(self,Date):
        for i in range(1,len(Date)):
            self.expectedOutput.append(Date[i])
        print(len(Date[:len(Date)-1]),len(self.expectedOutput))

        return

    #normalise dataset
    def convertDateToMS(self,DateList):        
        for i in range(len(DateList)):            
            DateList[i]=float(pd.to_datetime(DateList[i]).to_datetime64())/self.global_date_max
        return DateList
    
    def loadData(self):
        #import from csv
        
        with open(self.CSV_PATH) as f1:
            row_cnt=sum(1 for row in f1)
            row_cnt-=1

        print('row cnt:',row_cnt)

        #using pandas to retrieve csv data
        self.temp_data=pd.read_csv(self.CSV_PATH,usecols=self.colNames,header=0)
        self.temp_data['Date']=pd.to_datetime(self.temp_data['Date'])
        self.temp_data['Date']=self.temp_data['Date'].dt.year
        print(self.temp_data.columns,)
        self.temp_data=self.temp_data.groupby(self.temp_data['Date'],as_index=False).count()
        #self.temp_data=self.temp_data.reset_index()
        print(self.temp_data.columns)
        print(self.temp_data,self.temp_data.dtypes)
        print(self.temp_data.shape)
        pd.to_numeric(self.temp_data['Date'])
        pd.to_numeric(self.temp_data['Magnitude'])
        self.temp_data['Date']=self.temp_data.Date.astype(float)
        self.temp_data['Magnitude']=self.temp_data.Magnitude.astype(float)
        print(self.temp_data.dtypes)
        #for i in range(row_cnt//self.BATCH_SIZE):
        #    self.datasetList.append(tf.contrib.data.make_csv_dataset(self.CSV_PATH,batch_size=self.BATCH_SIZE))
        #self.dataset1=tf.contrib.data.make_csv_dataset(self.CSV_PATH,batch_size=row_cnt)
        
        #print(dataset1,type(dataset1))
        #iter=self.dataset1.make_one_shot_iterator()
        #self.next=iter.get_next()
        
        #print(next,type(next))
        #self.date,self.magnitude=self.next['Date'],self.next['Magnitude']
        
        #print(self.next['Date'],tf.size(self.date),type(self.next['Date']))
        #print(self.next['Magnitude'],tf.size(self.magnitude),type(self.next['Magnitude']))
        #with tf.Session() as sess:
        #    print(sess.run([self.date,self.magnitude,tf.size(self.date),tf.size(self.magnitude)]))            
        #self.magnitude=tf.manip.reshape(self.magnitude,(row_cnt,1))
        #self.date=tf.manip.reshape(self.date,(row_cnt,1))
        #print('shape:',self.magnitude.shape)
        #with tf.Session() as sess:
        #    print(sess.run(self.magnitude))

        self.dataset1=tf.convert_to_tensor(self.temp_data)
        #self.dataset2=tf.convert_to_tensor(self.temp_data['Magnitude'])
        

        print('Tensor:',self.dataset1)
        with tf.Session() as ses:
            print(ses.run(self.dataset1))

        return

    def dispData(self):
        x=np.random.sample((100,2))
        dataset2=tf.data.Dataset.from_tensor_slices(x)
        iter2=dataset2.make_one_shot_iterator()
        el=iter2.get_next()
        with tf.Session() as ses:
            print('test',ses.run(el))

        print(x)
        return

    def Initiate(self):
        print('in initiate')
        

            
        Wf=tf.Variable(np.random.rand(1,52),dtype=tf.float64)
        Wi=tf.Variable(np.random.rand(1,52),dtype=tf.float64)
        Wo=tf.Variable(np.random.rand(1,52),dtype=tf.float64)
        bf=tf.Variable(np.random.rand(1,2),dtype=tf.float64)
        bi=tf.Variable(np.random.rand(1,2),dtype=tf.float64)
        bo=tf.Variable(np.random.rand(1,2),dtype=tf.float64)
        #with tf.Session() as sess:
        #    sess.run(tf.global_variables_initializer())
        #    print(sess.run(Wf))

        #Forget gate - start cell
        forget_gate=tf.sigmoid((tf.matmul(Wf,self.dataset1)+bo))

        #Input gate - start cell
        input_gate=tf.sigmoid((tf.matmul(Wi,self.dataset1)+bi))
        C_t=tf.tanh((tf.matmul(Wi,self.dataset1)+bi))

        #output gate - start cell
        output_gate=tf.sigmoid((tf.matmul(Wo,self.dataset1)+bo))

        print(type(input_gate))
        with tf.Session() as ses:
            ses.run(tf.global_variables_initializer())
            print(ses.run(forget_gate))
            print(ses.run(input_gate))
            print(ses.run(C_t))
            print(ses.run(output_gate))
        pass


    pass



def main():
    obj=EarthQuakePhase1()
    obj.loadData()
    #obj.dispData()
    obj.Initiate()
    return

if __name__=='__main__':
    main()

