from Earthquake_phase_1_module3 import EarthQuakePhase1
import numpy as np
import time
import random as rd
import pandas as pd
from matplotlib import pyplot as plt
import pickle
import os.path

class LSTM(object):
    """description of class"""
    def __init__(self, *args, **kwargs):
        np.random.seed(int(time.time()))
        #initialise weights
        #initially set forget gate to 1 to remember everything
        #using only one weight list instead of W and U
        #Weights for Reset Gate.
        self.Weights_R_1=np.array(1)     #as per Felix lstm research to remember everything initially
        self.Weights_R_2=np.array(1)     #as per Felix lstm research to remember everything initially 
        # weights initialised between -0.1 to 0.1 as per Felix research
        #self.Weights_input_1=np.array(np.random.uniform(-0.1,0.1))
        #self.Weights_input_2=np.array(np.random.uniform(-0.1,0.1))
        #self.Weights_output_1=np.array(np.random.uniform(-0.1,0.1))
        #self.Weights_output_2=np.array(np.random.uniform(-0.1,0.1))
        #self.Weights_A_1=np.array(np.random.uniform(-0.1,0.1))
        #self.Weights_A_2=np.array(np.random.uniform(-0.1,0.1))

        #Weights for Update Gate.
        self.Weights_Z_1=np.array(np.random.random())
        self.Weights_Z_2=np.array(np.random.random())
        #Weights for Activation or memory gate
        self.Weights_S_1=np.array(np.random.random())
        self.Weights_S_2=np.array(np.random.random())
        #self.Weights_A_1=np.array(np.random.random())
        #self.Weights_A_2=np.array(np.random.random())

        #weight gradients
        self.gradient_weights_R_1=np.array(0)     
        self.gradient_weights_R_2=np.array(0)        
        self.gradient_weights_Z_1=np.array(0)
        self.gradient_weights_Z_2=np.array(0)
        self.gradient_weights_S_1=np.array(0)
        self.gradient_weights_S_2=np.array(0)
        #self.gradient_weights_A_1=np.array(0)
        #self.gradient_weights_A_2=np.array(0)

        self.gradient_R_bias=np.array(0)
        self.gradient_Z_bias=np.array(0)        
        #self.gradient_input_A_gate_bias=np.array(0)
        self.gradient_S_bias=np.array(0)

        #initialise biases
        #input and output gate bias should be -ve
        ##initially set forget gate to 1 to remember everything
        # as per felix, initialise input bias=0, forget bias=-2 and output bias=2
        #self.input_gate_bias=np.array(0)
        #self.forget_gate_bias=np.array(-2)
        #self.input_A_gate_bias=np.array(np.random.uniform(-0.1,0.1))
        #self.output_gate_bias=np.array(2)

        self.R_bias=np.array(1)
        self.Z_bias=np.array(np.random.random())        
        #self.input_A_gate_bias=np.array(np.random.random())
        self.S_bias=np.array(np.random.random())
        #initialise weight matrix. Comprises of all weights(Z)
        self.weights_matrix=np.zeros((3,3))
        self.weights_matrix[0][0]=self.Weights_Z_1; self.weights_matrix[0][1]=self.Weights_Z_2; self.weights_matrix[0][2]=self.Z_bias;
        self.weights_matrix[1][0]=self.Weights_S_1; self.weights_matrix[1][1]=self.Weights_S_2; self.weights_matrix[1][2]=self.S_bias;
        self.weights_matrix[2][0]=self.Weights_R_1; self.weights_matrix[2][1]=self.Weights_R_2; self.weights_matrix[2][2]=self.R_bias;
        #self.weights_matrix[3][0]=self.Weights_output_1; self.weights_matrix[3][1]=self.Weights_output_2; self.weights_matrix[3][2]=self.output_gate_bias;
        
        #print(self.weights_matrix)
        #gates initialisation
        self.R_gate=[]
        self.Z_gate=[]        
        #self.input_A_gate=[]
        self.S_gate=[]
        #self.cell_state=[]
        self.cell_output_previous=np.array(0)
        #self.cell_state_current=np.array(0)
        self.deltas=[]
        self.input=0
        #self.previous_output=np.array(0)
        self.cell_output=np.array(0)
        self.expected_cell_output=np.array(0)
        self.error=np.array(0)

        ####################################### Deltas for W ########################
        #self.delta_cell_state=np.array(0)
        #self.delta_input_A_gate=np.array(0)
        self.delta_Z_gate_W=np.array(0)
        self.delta_S_gate_W=np.array(0)
        self.delta_R_gate_W=np.array(0)

        self.Xt=np.zeros((1,3))
        self.Xt[0][0]=self.input
        self.Xt[0][1]=self.input
        self.Xt[0][2]=np.dot(self.Weights_S_2,self.input)

        self.delta_gates_W=np.zeros((1,3))
        self.delta_gates_W[0][0]=self.delta_Z_gate_W
        self.delta_gates_W[0][1]=self.delta_S_gate_W
        self.delta_gates_W[0][2]=self.delta_R_gate_W
        ####################################################################

        ####################################### Deltas for Wrec ########################
        #self.delta_cell_state=np.array(0)        
        #self.delta_input_A_gate=np.array(0)
        self.delta_Z_gate_Wrec=np.array(0)
        self.delta_S_gate_Wrec=np.array(0)
        self.delta_R_gate_Wrec=np.array(0)

        self.Yt=np.zeros((1,3))
        self.Yt[0][0]=self.cell_output_previous
        self.Yt[0][1]=self.cell_output_previous
        self.Yt[0][2]=np.dot(self.Weights_S_2,self.cell_output_previous)

        self.delta_gates_Wrec=np.zeros((1,3))
        self.delta_gates_Wrec[0][0]=self.delta_Z_gate_Wrec
        self.delta_gates_Wrec[0][1]=self.delta_S_gate_Wrec
        self.delta_gates_Wrec[0][2]=self.delta_R_gate_Wrec
        ####################################################################

        ###################################### Deltas for b #############################################
        #delta gates for b is same as for W

        self.Vt=np.zeros((1,3))
        self.Vt[0][0]=np.array(1)
        self.Vt[0][1]=np.array(1)
        self.Vt[0][2]=self.Weights_S_2
        ######################################################################

        ################################################ Deltas for cumulative error calculations #####################
        self.delta_Z_gate_error=np.array(0)
        self.delta_output_gate_error=np.array(0)
        self.delta_S_gate_error=np.array(0)
        self.delta_R_gate_error=np.array(0)

        self.U=np.zeros((4,1))
        self.U[0][0]=self.Weights_Z_2
        self.U[1][0]=1
        self.U[2][0]=self.Weights_S_2
        self.U[3][0]=np.dot(self.Weights_S_2,self.Weights_R_2)

        self.delta_gates_cumulative_error=np.zeros((1,4))
        self.delta_gates_cumulative_error[0][0]=self.delta_Z_gate_error
        self.delta_gates_cumulative_error[0][1]=self.delta_output_gate_error
        self.delta_gates_cumulative_error[0][2]=self.delta_S_gate_error
        self.delta_gates_cumulative_error[0][3]=self.delta_R_gate_error

        ################################################################################

        self.delta_cell_output=np.array(0)
        self.delta_cumulative_error=np.array(0)

        #input matrix
        self.input_matrix=np.zeros(3)

        self.input_matrix_X=np.zeros((1,1))
        self.input_matrix_U=np.zeros((1,1))
        self.input_matrix_b=np.zeros((1,1))

        #variables needed for weights calculation
        self.delta_gate_list=np.zeros((4,1))
        return super().__init__(*args, **kwargs)
        
    pass

class LSTMNetwork():
    def __init__(self, cells,data,global_date_max):
        self.lstmCellsCnt=cells
        self.dataset=data
        self.global_date_max=global_date_max
        #self.expectedOutput=expectedOutput
        self.lstmCellObjList=[]
        self.lastPrediction=np.array(0)
        self.predictionResult=[]
        self.lastPredictionSet=[0 for i in range(cells)]
        return 

    def initialiseLSTMCells(self):
        for i in range(self.lstmCellsCnt):
            self.lstmCellObjList.append(LSTM())

    def TrainNetwork(self):
        listPtr=0
        lastPrediction=np.array(0)
        temp_previous_output=np.array(0)
        temp_previous_cell_state=np.array(0)
        listPtr=0
        for cntr in range(107):#12000
            listPtr=0
            print('epoch# ',cntr,' running...')
            #for i in range(len(self.dataset)-1-self.lstmCellsCnt+1):
                #print('set#: ',i+1)
                #cellDataSet=self.dataset[listPtr:listPtr+self.lstmCellsCnt+1]

            while listPtr<len(self.dataset)-1:

                cellDataSet=self.dataset[listPtr:listPtr+(2)] #3 cause 2 lstm cells
                listPtr += 1
                temp_previous_output = np.array(0)
                temp_previous_cell_state = np.array(0)
                #if len(cellDataSet)<(self.lstmCellsCnt+1):
                #    break

                for j in range(self.lstmCellsCnt):
                    #if j==0:
                    #    print('prev cell state:',temp_previous_cell_state)
                    lstmCell = self.lstmCellObjList[j]
                    lstmCell.expected_cell_output = cellDataSet[1]
                                
                    lstmCell.previous_output = temp_previous_output           #initializing previous cell output
                    #lstmCell.cell_state_previous=temp_previous_cell_state       #initializing previous cell state
                    lstmCell.input=cellDataSet[0]
                    temp_previous_output = self.forwardPass(lstmCell,cellDataSet[0])    #forward pass

                    lstmCell.error = temp_previous_output-lstmCell.expected_cell_output    #error calculation
                    
                    #lstmCell.cell_state_current=temp_previous_cell_state        #current cell state
                    #print('error with cell# ',j+1,':',lstmCell.error,'\t input: ',cellDataSet[0]*15000,' \tpredicted:',temp_previous_output*15000)
                    #print('error with cell# ',j+1,':',lstmCell.error,'\t input: ',pd.to_datetime(cellDataSet[j]*self.global_date_max),' \tpredicted:',pd.to_datetime(temp_previous_output*self.global_date_max))
                    lastPrediction = temp_previous_output
                    self.lstmCellObjList[j] = lstmCell
                    if j<self.lstmCellsCnt-1:
                        self.lastPredictionSet[j] = lastPrediction

            self.backPropagate(self.lstmCellObjList,0)
            self.updateWeights(self.lstmCellObjList)
            
        return lastPrediction

    def updateWeights(self,lstmCellObjList):
        total_delta_W=np.zeros((1,3))
        total_delta_U=np.zeros((1,3))
        total_delta_b=np.zeros((1,3))
        total_delta=np.zeros((3,3))
        learning_rate=0.85 #0.85
        for i in range(len(lstmCellObjList)):
            lstmCellObj=lstmCellObjList[i]
            total_delta_W+=np.multiply(lstmCellObj.delta_gates_W ,lstmCellObj.Xt)
            total_delta_U+=np.multiply(lstmCellObj.delta_gates_Wrec,lstmCellObj.Yt)
            #if i==0:
            #    total_delta_U+=np.array(0)                
            #else:
            #    total_delta_U+=np.dot(lstmCellObj.delta_gate_list,lstmCellObjList[i-1].input_matrix_U)

            total_delta_b+=np.multiply(lstmCellObj.delta_gates_W , lstmCellObj.Vt)

            total_delta[0]=total_delta_W
            total_delta[1]=total_delta_U
            total_delta[2]=total_delta_b

            #lstmCellObj.weights_matrix=lstmCellObj.weights_matrix-np.dot(learning_rate,total_delta.T)

        for i in range(len(lstmCellObjList)):
            lstmCellObj=lstmCellObjList[i]
            lstmCellObj.weights_matrix = lstmCellObj.weights_matrix - np.dot(learning_rate,total_delta)
            
        return


    def backPropagate(self,lstmCellObjList,cellLocation):
        for cellLoc in range(len(lstmCellObjList)-1,-1,-1):     #back propagate the cell just forward propagated.
            lstmCellObj=lstmCellObjList[cellLoc]
            #condition for last cell in the series of LSTM network
            if cellLoc==len(lstmCellObjList)-1:
                backtrackObj=LSTM()
            else:
                backtrackObj=lstmCellObjList[cellLoc+1]
            
            #delta cell output
            if cellLoc==len(lstmCellObjList)-1:
                lstmCellObj.delta_cell_output = lstmCellObj.error+np.array(0)
            else:                
                lstmCellObj.delta_cell_output = lstmCellObj.error+lstmCellObj.delta_cumulative_error

            ######### cumulative error calculation
            lstmCellObj.delta_Z_gate_error = np.dot(lstmCellObj.delta_cell_output,np.dot((lstmCellObj.S_gate - lstmCellObj.cell_output_previous),np.dot(lstmCellObj.Z_gate,(np.array(1)-lstmCellObj.Z_gate))))
            lstmCellObj.delta_output_gate_error = np.dot(lstmCellObj.delta_cell_output, (np.array(1)-lstmCellObj.Z_gate))
            lstmCellObj.delta_S_gate_error = np.dot(lstmCellObj.delta_cell_output,np.dot(lstmCellObj.Z_gate,np.dot(lstmCellObj.R_gate,np.array(1)-np.square(lstmCellObj.S_gate))))
            lstmCellObj.delta_R_gate_error = np.dot(lstmCellObj.delta_cell_output,np.dot(lstmCellObj.Z_gate,np.dot(np.array(1)-np.square(lstmCellObj.S_gate),np.dot(lstmCellObj.cell_output_previous,np.dot(lstmCellObj.R_gate,np.array(1)-lstmCellObj.R_gate)))))

            lstmCellObj.U[0][0] = lstmCellObj.Weights_Z_2
            lstmCellObj.U[1][0] = 1
            lstmCellObj.U[2][0] = lstmCellObj.Weights_S_2
            lstmCellObj.U[3][0] = np.dot(lstmCellObj.Weights_S_2,lstmCellObj.Weights_R_2)
                        
            lstmCellObj.delta_gates_cumulative_error[0][0] = lstmCellObj.delta_Z_gate_error
            lstmCellObj.delta_gates_cumulative_error[0][1] = lstmCellObj.delta_output_gate_error
            lstmCellObj.delta_gates_cumulative_error[0][2] = lstmCellObj.delta_S_gate_error
            lstmCellObj.delta_gates_cumulative_error[0][3] = lstmCellObj.delta_R_gate_error

            ########## delta gate  for W
            lstmCellObj.delta_Z_gate_W = lstmCellObj.delta_Z_gate_error
            lstmCellObj.delta_S_gate_W = np.dot(lstmCellObj.delta_cell_output,np.dot(lstmCellObj.Z_gate,np.array(1)-np.square(lstmCellObj.S_gate)))
            lstmCellObj.delta_R_gate_W = lstmCellObj.delta_R_gate_error

            lstmCellObj.Xt[0][0] = lstmCellObj.input
            lstmCellObj.Xt[0][1] = lstmCellObj.input
            lstmCellObj.Xt[0][2] = np.dot(lstmCellObj.input,lstmCellObj.Weights_S_2)

            lstmCellObj.delta_gates_W[0][0] = lstmCellObj.delta_Z_gate_W
            lstmCellObj.delta_gates_W[0][1] = lstmCellObj.delta_S_gate_W
            lstmCellObj.delta_gates_W[0][2] = lstmCellObj.delta_R_gate_W

            ########## delta gate for Wrec
            lstmCellObj.delta_Z_gate_Wrec = lstmCellObj.delta_Z_gate_error
            lstmCellObj.delta_S_gate_Wrec = lstmCellObj.delta_S_gate_error
            lstmCellObj.delta_R_gate_Wrec = lstmCellObj.delta_R_gate_error

            lstmCellObj.Yt[0][0] = lstmCellObj.cell_output_previous
            lstmCellObj.Yt[0][1] = lstmCellObj.cell_output_previous
            lstmCellObj.Yt[0][2] = np.dot(lstmCellObj.cell_output_previous,lstmCellObj.Weights_S_2)

            lstmCellObj.delta_gates_Wrec[0][0] = lstmCellObj.delta_Z_gate_Wrec
            lstmCellObj.delta_gates_Wrec[0][1] = lstmCellObj.delta_S_gate_Wrec
            lstmCellObj.delta_gates_Wrec[0][2] = lstmCellObj.delta_R_gate_Wrec  

            ######### delta gate for b
            lstmCellObj.Vt[0][0]=np.array(1)
            lstmCellObj.Vt[0][1]=np.array(1)
            lstmCellObj.Vt[0][2]=lstmCellObj.Weights_S_2

            if cellLoc>0:
                lstmCellObjList[cellLoc-1].delta_cumulative_error = np.dot(lstmCellObj.delta_gates_cumulative_error,lstmCellObj.U)
            
                
            #####################################################################################################
                
                

        #    ##delta cell state
        #    #if cellLoc==len(lstmCellObjList)-1:
        #    #    lstmCellObj.delta_cell_state=np.dot(lstmCellObj.delta_cell_output,np.dot(lstmCellObj.output_gate,np.array(1)-np.square(np.tanh(lstmCellObj.cell_state_current))))+np.array(0)
        #    #else:
        #    #    lstmCellObj.delta_cell_state=np.dot(lstmCellObj.delta_cell_output,np.dot(lstmCellObj.output_gate,np.array(1)-np.square(np.tanh(lstmCellObj.cell_state_current))))+np.dot(backtrackObj.delta_cell_state,backtrackObj.forget_gate)
                       
        #    #delta input Activation gate
        #    lstmCellObj.delta_input_A_gate=np.dot(lstmCellObj.delta_cell_state,np.dot(lstmCellObj.input_gate,np.array(1)-np.square(lstmCellObj.input_A_gate)))

        #    #delta input gate
        #    lstmCellObj.delta_input_gate=np.dot(lstmCellObj.delta_cell_state,np.dot(lstmCellObj.input_A_gate,np.dot(lstmCellObj.input_gate,np.array(1)-lstmCellObj.input_gate)))

        #    #delta forget gate
        #    if cellLoc==0:
        #        lstmCellObj.delta_forget_gate=0.0
        #    else:
        #        lstmCellObj.delta_forget_gate=np.dot(lstmCellObj.delta_cell_state,np.dot(lstmCellObj.cell_state_previous,np.dot(lstmCellObj.forget_gate,np.array(1)-lstmCellObj.forget_gate)))

        #    #delta output gate
        #    lstmCellObj.delta_output_gate=np.dot(lstmCellObj.delta_cell_output,np.dot(np.tanh(lstmCellObj.cell_state_current),np.dot(lstmCellObj.output_gate,np.array(1)-lstmCellObj.output_gate)))

        #    #delta cumulative error
        #    if cellLoc==len(lstmCellObjList)-1:
        #        lstmCellObj.delta_cumulative_error=np.array(0)

        ##    self.weights_matrix=np.zeros((4,3))
        ##self.weights_matrix[0][0]=self.Weights_A_1; self.weights_matrix[0][1]=self.Weights_A_2; self.weights_matrix[0][2]=self.input_A_gate_bias;
        ##self.weights_matrix[1][0]=self.Weights_input_1; self.weights_matrix[1][1]=self.Weights_input_2; self.weights_matrix[1][2]=self.input_gate_bias;
        ##self.weights_matrix[2][0]=self.Weights_forget_1; self.weights_matrix[2][1]=self.Weights_forget_2; self.weights_matrix[2][2]=self.forget_gate_bias;
        ##self.weights_matrix[3][0]=self.Weights_output_1; self.weights_matrix[3][1]=self.Weights_output_2; self.weights_matrix[3][2]=self.output_gate_bias;
        
        #    lstmCellObj.delta_gate_list[0][0] = lstmCellObj.delta_input_A_gate
        #    lstmCellObj.delta_gate_list[1][0] = lstmCellObj.delta_input_gate
        #    lstmCellObj.delta_gate_list[2][0] = lstmCellObj.delta_forget_gate
        #    lstmCellObj.delta_gate_list[3][0] = lstmCellObj.delta_output_gate
            
        #    if cellLoc>0:
        #        weight_U_matrix = np.zeros((1,4))
        #        weight_U_matrix[0][0] = lstmCellObj.weights_matrix[0][1]  #lstmCellObj.Weights_A_2
        #        weight_U_matrix[0][1] = lstmCellObj.weights_matrix[1][1]   #lstmCellObj.Weights_input_2
        #        weight_U_matrix[0][2] = lstmCellObj.weights_matrix[2][1]  #lstmCellObj.Weights_forget_2
        #        weight_U_matrix[0][3] = lstmCellObj.weights_matrix[3][1]  #lstmCellObj.Weights_output_2

                
                
        #        lstmCellObjList[cellLoc-1].delta_cumulative_error = np.dot(weight_U_matrix,lstmCellObj.delta_gate_list)
        #        #lstmCellObj.delta_cumulative_error=np.dot(weight_U_matrix,lstmCellObj.delta_gate_list)
            
        return

    def forwardPass(self,lstmCell,current_input):
        #assemble input matrix        
        lstmCell.input_matrix = np.hstack((current_input,lstmCell.previous_output,1))
        lstmCell.input_matrix_X[0] = current_input
        lstmCell.input_matrix_U[0] = lstmCell.previous_output
        lstmCell.input_matrix_b[0] = 1
        

        #Reset gate computation (R)
        complete_weights = np.hstack((lstmCell.weights_matrix[2][0],lstmCell.weights_matrix[2][1],lstmCell.weights_matrix[2][2]))
        complete_weights = complete_weights.reshape(complete_weights.shape[0],-1)           
        lstmCell.R_gate = self.sigmoid(np.dot(complete_weights.T,lstmCell.input_matrix))
                
        #input Activation gate computation (S)
        complete_weights=np.hstack((lstmCell.weights_matrix[1][0],np.dot(lstmCell.weights_matrix[1][1],lstmCell.R_gate),lstmCell.weights_matrix[1][2]))
        complete_weights=complete_weights.reshape(complete_weights.shape[0],-1)
        lstmCell.S_gate =np.tanh(np.dot(complete_weights.T,lstmCell.input_matrix))

        #Update gate computation (Z)
        complete_weights=np.hstack((lstmCell.weights_matrix[0][0],lstmCell.weights_matrix[0][1],lstmCell.weights_matrix[0][2]))
        complete_weights=complete_weights.reshape(complete_weights.shape[0],-1)
        lstmCell.Z_gate =self.sigmoid(np.dot(complete_weights.T,lstmCell.input_matrix))
                
        #cell output computation
        lstmCell.cell_output = np.dot((np.array(1)-lstmCell.Z_gate),lstmCell.previous_output) + np.dot(lstmCell.Z_gate,lstmCell.S_gate)       #np.array(np.dot(lstmCell.output_gate,np.tanh(lstmCell.cell_state)))
        
        return np.array(lstmCell.cell_output)

    def sigmoid(self,inX):
        return 1/(1+np.exp(-inX))

    def predict(self,current_input,test_data):
        test_data=list(test_data)
        cnt=0
        i=0
        slidingPtr=0
        pred_val=np.array(0)
        prev_val=np.array(0)
        print('size: ',len(self.lstmCellObjList))
        threshold=float((pd.to_datetime('12/31/2018')).to_datetime64())/self.global_date_max
        cellCnt=len(self.lstmCellObjList)
        #while(pred_val<=threshold):
        while(slidingPtr<len(test_data)):
            #if current_input>=20:          #12312022:
            #    break
            prev_val=np.array(0)
            inputList=list(test_data[slidingPtr:slidingPtr+1]) #test_data[slidingPtr]  #list(test_data[slidingPtr:cellCnt+slidingPtr])
            #print('current input:',pd.to_datetime(current_input*self.global_date_max))
            #if len(inputList)<cellCnt:
            #    break
            prev_val=np.array(0)
            for cnt in range(len(self.lstmCellObjList)):
                #assemble input matrix    
                lstmCell=self.lstmCellObjList[cnt]
                
                lstmCell.previous_output=prev_val
                #print('prev output:',lstmCell.previous_output)
                #lstmCell.input_matrix=np.hstack((current_input,lstmCell.previous_output,1))
                lstmCell.input_matrix = np.hstack((inputList[0] ,lstmCell.previous_output,1))
                lstmCell.input_matrix_X[0] = inputList[0]
                lstmCell.input_matrix_U[0] = lstmCell.previous_output
                lstmCell.input_matrix_b[0] = 1
        

                #Reset gate computation (R)
                complete_weights = np.hstack((lstmCell.weights_matrix[2][0],lstmCell.weights_matrix[2][1],lstmCell.weights_matrix[2][2]))
                complete_weights = complete_weights.reshape(complete_weights.shape[0],-1)           
                lstmCell.R_gate = self.sigmoid(np.dot(complete_weights.T,lstmCell.input_matrix))
                
                #input Activation gate computation (S)
                complete_weights=np.hstack((lstmCell.weights_matrix[1][0],np.dot(lstmCell.weights_matrix[1][1],lstmCell.R_gate),lstmCell.weights_matrix[1][2]))
                complete_weights=complete_weights.reshape(complete_weights.shape[0],-1)
                lstmCell.S_gate =np.tanh(np.dot(complete_weights.T,lstmCell.input_matrix))

                #Update gate computation (Z)
                complete_weights=np.hstack((lstmCell.weights_matrix[0][0],lstmCell.weights_matrix[0][1],lstmCell.weights_matrix[0][2]))
                complete_weights=complete_weights.reshape(complete_weights.shape[0],-1)
                lstmCell.Z_gate =self.sigmoid(np.dot(complete_weights.T,lstmCell.input_matrix))
                
                #cell output computation
                lstmCell.cell_output = np.dot((np.array(1)-lstmCell.Z_gate),lstmCell.previous_output) + np.dot(lstmCell.Z_gate,lstmCell.S_gate)       #np.array(np.dot(lstmCell.output_gate,np.tanh(lstmCell.cell_state)))
        

                #previous output
                prev_val=lstmCell.cell_output
                pred_val=lstmCell.cell_output
                #prediction
                #print('input:',pd.to_datetime(inputList[0]*self.global_date_max),' predicted:',pd.to_datetime(lstmCell.cell_output*self.global_date_max))
                #self.predictionResult.append(pred_val)

            print('#'*100)
            print('input:',pd.to_datetime(inputList[0]*self.global_date_max),'total prediction:',pd.to_datetime(pred_val*self.global_date_max),' pred:',pred_val,' threshold:',threshold,' length:',len(test_data))
            #print('total prediction:',pd.to_datetime(pred_val*15000))
            
            #################################### Normalise datetime #######################################
            #normalisePrediction = pd.to_datetime(pred_val*self.global_date_max).normalize()
            ##print(normalisePrediction)
            #pred_val=float(pd.to_datetime(normalisePrediction[0]).to_datetime64())/self.global_date_max
            
            #####################################################################################

            self.predictionResult.append(pred_val)
            #test_data.append(pred_val)
            print('*'*100)
            i+=1
            current_input=pred_val
            slidingPtr+=1
            #if slidingPtr==len(test_data):
            #    test_data.append(pred_val)

        print('while loop done:',len(self.predictionResult))
        x=input('start checking...........')
        while pred_val<threshold and 1==0:
            prev_val=np.array(0)
            inputList=list([pred_val]) #test_data[slidingPtr]  #list(test_data[slidingPtr:cellCnt+slidingPtr])
            #print('current input:',pd.to_datetime(current_input*self.global_date_max))
            #if len(inputList)<cellCnt:
            #    break
            prev_val=np.array(0)
            temp_val=np.array(0)
            for cnt in range(len(self.lstmCellObjList)):
                #assemble input matrix    
                lstmCell=self.lstmCellObjList[cnt]
                
                lstmCell.previous_output=prev_val
                #print('prev output:',lstmCell.previous_output)
                #lstmCell.input_matrix=np.hstack((current_input,lstmCell.previous_output,1))
                lstmCell.input_matrix = np.hstack((current_input,lstmCell.previous_output,1))
                lstmCell.input_matrix_X[0] = current_input
                lstmCell.input_matrix_U[0] = lstmCell.previous_output
                lstmCell.input_matrix_b[0] = 1
        

                #Reset gate computation (R)
                complete_weights = np.hstack((lstmCell.weights_matrix[2][0],lstmCell.weights_matrix[2][1],lstmCell.weights_matrix[2][2]))
                complete_weights = complete_weights.reshape(complete_weights.shape[0],-1)           
                lstmCell.R_gate = self.sigmoid(np.dot(complete_weights.T,lstmCell.input_matrix))
                
                #input Activation gate computation (S)
                complete_weights=np.hstack((lstmCell.weights_matrix[1][0],np.dot(lstmCell.weights_matrix[1][1],lstmCell.R_gate),lstmCell.weights_matrix[1][2]))
                complete_weights=complete_weights.reshape(complete_weights.shape[0],-1)
                lstmCell.S_gate =np.tanh(np.dot(complete_weights.T,lstmCell.input_matrix))

                #Update gate computation (Z)
                complete_weights=np.hstack((lstmCell.weights_matrix[0][0],lstmCell.weights_matrix[0][1],lstmCell.weights_matrix[0][2]))
                complete_weights=complete_weights.reshape(complete_weights.shape[0],-1)
                lstmCell.Z_gate =self.sigmoid(np.dot(complete_weights.T,lstmCell.input_matrix))
                
                #cell output computation
                lstmCell.cell_output = np.dot((np.array(1)-lstmCell.Z_gate),lstmCell.previous_output) + np.dot(lstmCell.Z_gate,lstmCell.S_gate)       #np.array(np.dot(lstmCell.output_gate,np.tanh(lstmCell.cell_state)))

                #previous output
                prev_val=lstmCell.cell_output
                temp_val=lstmCell.cell_output
                #prediction
                #print('input:',pd.to_datetime(inputList[0]*self.global_date_max),' predicted:',pd.to_datetime(lstmCell.cell_output*self.global_date_max))
                #self.predictionResult.append(pred_val)
            pred_val=temp_val
            print('#'*100)
            print('input:',pd.to_datetime(inputList[0]*self.global_date_max),'future total prediction:',pd.to_datetime(pred_val*self.global_date_max),' pred:',pred_val,' threshold:',threshold,' length:',len(test_data))
            x=input('Next:')
        
        return 


    pass


obj=EarthQuakePhase1()
data,global_date_max,test_data=obj.loadData_LSTM()

### custom code
#global_date_max=1
#x=np.arange(0,50,0.1)
#x=x.reshape(x.shape[0],-1)
#x=np.sin(x)
#data=x[:400]
#test_data=x[400:]
#lstmNet=LSTMNetwork(5,data,global_date_max) #25

#if os.path.isfile('sineTraining3.sb'):
#    with open('sineTraining.sb','rb') as lstmRead:
#        lstmNet=pickle.load(lstmRead)
#else:
#    print('would init')
#    lstmNet.initialiseLSTMCells()
#    lstmNet.lastPrediction=lstmNet.TrainNetwork()
#    #lstmNet.lastPrediction=lstmNet.TrainNetwork()

##save status of training in a file
#with open('sineTraining.sb','wb') as lstmWrite:
#    pickle.dump(lstmNet,lstmWrite)


### end custom code

#data=np.random.randint(1,15000,size=(21000,1))
#data.sort(axis=0)
#test_data=data[20000:]
#data=data[:20000]
#data=np.divide(data,15000)
#test_data=np.divide(test_data,15000)
#print(len(data),len(data)-10+1)
#print('input: ',data)
print('-'*100)
#lstmNet=LSTMNetwork(5,data,1)

###correct code
lstmNet=LSTMNetwork(5,data,global_date_max) #25

if os.path.isfile('lstmTraining.bv'):
    with open('lstmTraining.bv','rb') as lstmRead:
        lstmNet=pickle.load(lstmRead)
else:
    print('would init')
    lstmNet.initialiseLSTMCells()
    lstmNet.lastPrediction=lstmNet.TrainNetwork()
    #lstmNet.lastPrediction=lstmNet.TrainNetwork()

#save status of training in a file
with open('lstmTraining.bv','wb') as lstmWrite:
    pickle.dump(lstmNet,lstmWrite)
###correct code ends

print('*'*100)
#print('total prediction:',pd.to_datetime(lstmNet.lastPrediction*lstmNet.global_date_max))
lstmNet.predict(lstmNet.lastPrediction,list(data)+list(test_data)) #list(data)+list(test_data)
#lstmNet.predict(lstmNet.lastPrediction,list(test_data))
#print('-'*100)
#for i in range(20):
#    print(pd.to_datetime(test_data[i]*global_date_max))
print('final lenght:',len(list(data)+list(test_data)),':',len(lstmNet.predictionResult))
#plot graphs
plt.plot(list(data)+list(test_data),label='Training & Test Data')
plt.plot(list(data)+list(test_data),label='Test Data')

plt.plot(list(lstmNet.predictionResult),label='Predicted Data')
plt.legend()
plt.show(block=False)

plt.show()


