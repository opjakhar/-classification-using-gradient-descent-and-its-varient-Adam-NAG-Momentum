#----importing packages-------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import pickle
import argparse
np.random.seed(1234)

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float,dest="abc")
parser.add_argument("--momentum", type=float,dest="abc1")
parser.add_argument("--num_hidden", type=int,dest="abc2")
parser.add_argument("--sizes",dest="abc3")
parser.add_argument("--activation",dest="abc4")
parser.add_argument("--loss",dest="abc5")
parser.add_argument("--opt",dest="abc6")
parser.add_argument("--batch_size",type=int,dest="abc7")
parser.add_argument("--anneal",dest="abc8")
parser.add_argument("--save_dir", type=str,dest="abc9")
parser.add_argument("--expt_dir",type=str, dest="abc10")
parser.add_argument("--train", type=str,dest="abc11")
parser.add_argument("--test",type=str,dest="abc12")
parser.add_argument("--val",type=str, dest="abc13")
parser.add_argument("--pretrain",type=str, dest="abc14")
data_field= parser.parse_args()

#---------parameter information---------------------------------------
eta=data_field.abc                        			# learning rate
momentum=data_field.abc1              			# momentum
num_hidden=data_field.abc2		     		        # no. of hidden layers
sizes = [int(value) for value in data_field.abc3.split(',')]# nueron in each layer or size of hidden layer
sizes.append(10)  # adding output layer nueron or output layer size
activation_name = data_field.abc4
loss_name=data_field.abc5
opt=data_field.abc6
#print(opt)
batch_size=data_field.abc7
##print(batch_size)
anneal=data_field.abc8
#print(anneal)


train_file=data_field.abc11

test_file=data_field.abc12
val_file=data_field.abc13
model_file=data_field.abc9
log_file=data_field.abc10
pretrain=data_field.abc14

max_iteration=60
num_outputlayer=10
step_number=100

model_log_file=open(model_file+"model_paramtetr.pickle","w+")
f_train_log = open(log_file+"log_train.txt","w+")

f_val_log = open(log_file+"log_val.txt","w+")

log_train_string=""
log_val_string=""
anneal_previous_val_error=0


#-----------------------------------new code------------------------
#----ifunction declaration-------------------------------------------------------------
def initial_weight(col,num_hidden,h_size):
        W=dict()
        for c in range(num_hidden+1):
            W.update({c:np.array([[ np.random.randn() for i2 in range(h_size[c-1]) ] if c>0 else [ np.random.randn() for i2 in range(col)] for i1 in range(h_size[c]) ])})
        return W

def initial_bias(col,num_hidden,h_size):
    b=dict()
    c=0
    for c in range(num_hidden+1):
        print("loop run : {}".format(c))
        b.update({ c: np.array([ np.random.randn() for i1 in range(h_size[c]) ] ) })
        #print(b[c])
    return b
       
def f_norm(train_data):
    data_mean=np.mean(train_data,axis=0)
    data_std=np.std(train_data,axis=0)+0.0001
    train_data=((train_data - data_mean)/data_std)
    return train_data
   
def f_sigmoid(vector):
    act_func=1.0/(1.0+np.exp(-vector))
    return act_func

def f_tanh(vector):
    act_func=np.tanh(-vector)
    return act_func

def f_activation(ww,xx,bb):
   # print("W dimenson : {}".format(ww.shape))
   # print("x dimenson : {}".format(xx.shape))
   # print("b dimenson : {}".format(bb.shape))
    return np.dot(ww,xx)+bb

def f_softmax(pre_soft):
   # activation_length=vector.shape[1]
    acc=np.array(pre_soft)
    ar=np.exp(acc)
    s=ar.sum(axis=0, dtype='float')
    y_cap=(ar/s)
    return y_cap

def ce_loss(arr1,arr2):
    l_arr1=arr2.shape[1]
    cross_error=dict()
    for i in range(l_arr1):
        if l_arr1 == 1:
            true_prob=arr1
            pridict_prob=np.log(arr2[:,i])
            b_b_len=len(pridict_prob)
            pridict_prob.shape=(b_b_len,1)
            cross_error.update({i:-np.dot(true_prob,np.transpose(pridict_prob))})
        else:
            true_prob=np.array(arr1[:,i])
            pridict_prob=np.log(arr2[:,i])
            cross_error.update({i:-np.dot(true_prob,np.transpose(pridict_prob))})
    return cross_error

def sq_loss(arr1,arr2):
    arr=arr1-arr2
    arr=arr**2
    s=arr.sum()
    return s

def g_detch(activation_matrix):
    #res=1-np.multiply(f_sigmoid(activation_matrix),f_sigmoid(activation_matrix))
    res=np.multiply(f_sigmoid(activation_matrix),(1-f_sigmoid(activation_matrix)))
    return res

def forward_propogation(W,b,processing_data,num_hidden,activation_name):
    pre_activation = {}
    activation = {}
    j=0
    for j in range(num_hidden+1):
            if j==0:     
               
                pre_activation.update({j:(f_activation((W[j]),np.transpose(processing_data),np.array(b[j])))})
                if activation_name == "sigmoid":
                    activation.update({j:f_sigmoid(pre_activation[j])})
                if activation_name == "tanh":
                    activation.update({j:f_tanh(pre_activation[j])})
            elif j==num_hidden:
                pre_activation.update({j:(f_activation((W[j]),np.array((activation[j-1])),np.array(b[j])))})
                activation.update({j:f_softmax((pre_activation[j]))})
            else:
                pre_activation.update({j:(f_activation((W[j]),np.array(activation[j-1]),np.array(b[j])))})
                if activation_name == "sigmoid":
                    activation.update({j:f_sigmoid((pre_activation[j]))})
                if activation_name == "tanh":
                    activation.update({j:f_tanh((pre_activation[j]))})
    return W,b,pre_activation,activation


def backward_propogation(W,b,arr,num_hidden,pre_activation,activation,processing_data,loss_name):
    grad_activation={}
    grad_weight={}
    grad_pre_activation={}
    grad_bias={}
    if loss_name=='ce':
        grad_pre_activation.update({num_hidden:-(arr-activation[num_hidden])})
    else:
        grad_pre_activation.update({num_hidden:np.multiply((np.multiply((activation[num_hidden]-arr),activation[num_hidden])),(1-activation[num_hidden]))})
        
   # print(arr.shape)
   # print(activation[num_hidden].shape)
   # print(grad_pre_activation[num_hidden].shape)
    for g_k in range(num_hidden,0,-1):
   
        grad_weight.update({g_k:np.dot(grad_pre_activation[g_k],np.transpose(activation[(g_k - 1)]))})
        
        b_t_grad=grad_pre_activation[g_k]
        b_detch=b_t_grad.sum(axis=1)
        b_len=len(b_detch)
        b_detch.shape=(b_len,1)
        grad_bias.update({g_k:b_detch})
       
       

        grad_activation.update({(g_k-1):np.dot(np.transpose(W[g_k]),grad_pre_activation[g_k])})
       
        grad_pre_activation.update({(g_k - 1):np.multiply(grad_activation[(g_k - 1)],g_detch(pre_activation[g_k - 1]))})

      
    for g_k in range(num_hidden,0,-1):
        W[g_k] = W[g_k] - eta*grad_weight[g_k]
        b[g_k] = b[g_k] - eta*grad_bias[g_k]
       
    grad_weight.update({0:np.matmul(grad_pre_activation[0],processing_data)})
  
   
 
    b_t_grad=grad_pre_activation[0]
    b_detch=b_t_grad.sum(axis=1)
    b_len=len(b_detch)
    b_detch.shape=(b_len,1)
    grad_bias.update({0:b_detch})
    
 
   
    return grad_weight,grad_bias


def update_paramters_gd(W,b,grad_weight,grad_bias,eta):
    for g_k in range(num_hidden,0,-1):
        W[g_k] = W[g_k] - eta*grad_weight[g_k]
        b[g_k] = b[g_k] - eta*grad_bias[g_k]
        
    W[0] = W[0] - eta*grad_weight[0]
    b[0] = b[0] - eta*grad_bias[0]
    
    return W,b


def update_paramters_momentum(num_hidden,W,b,grad_weight,grad_bias,eta,momentum,previous_update_w,previous_update_b):
    
    for g_k in range(num_hidden,0,-1):
        temp=momentum*previous_update_w[g_k] + eta*grad_weight[g_k]
        W[g_k] = W[g_k] - previous_update_w[g_k]
        previous_update_w[g_k]=temp
        temp2=momentum*previous_update_b[g_k] + eta*grad_bias[g_k]
        b[g_k] = b[g_k] - previous_update_b[g_k]
        previous_update_b[g_k]=temp2
       
    temp=momentum*previous_update_w[0] + eta*grad_weight[0]
    W[0] = W[0] - previous_update_w[0]
    previous_update_w[0]=temp
    temp2=momentum*previous_update_b[0] + eta*grad_bias[0]
    b[0] = b[0] - previous_update_b[0]
    previous_update_b[0]=temp2
    return W,b,previous_update_w,previous_update_b


def accuracy_error_calculation(W,b,num_hidden,processing_data11,dummy_label,loss_name,activation_name):
    
    W,b,pre_activation,activation=forward_propogation(W,b,processing_data11,num_hidden,activation_name)
    
    ap=activation[num_hidden]
    y_cap_len=ap.shape[1]

    pridict_label=dict()
    
    #------loss calculation-------------
    
    for i in range(processing_data11.shape[0]):
        y_original=[0]*10
        y_original[dummy_label[i]]=1
        y_original=np.array(y_original)
        if i==0:
            arr=y_original
        else:
            arr= np.vstack([arr, y_original])

    arr=np.transpose(arr) 

    if loss_name=="sq":
        loss=sq_loss(arr,activation[num_hidden])
        total_loss=loss

    if loss_name=="ce":
        loss=ce_loss(arr,activation[num_hidden])
        total_loss=sum(loss.values())
        
    dict_loss_len=processing_data11.shape[0]
    total_loss=total_loss/dict_loss_len

    #-------------------
    for y_cp_l in range(y_cap_len):
        asa=ap[:,y_cp_l]
        pridict_label.update({y_cp_l:np.argmax(asa)})

    label_count=0
    y_cp_l=0
    for y_cp_l in range(y_cap_len):
        if pridict_label[y_cp_l] == dummy_label[y_cp_l]:      
            label_count+=1

    accuracy=(label_count/y_cap_len)*100
  #  print('accuracy={}'.format(accuracy))
    error=100-round(accuracy,2)
    
    return round(total_loss,2),accuracy,round(error,2)
    
    

    

#-----------loading data --------------------------------------------------------------
All_data=pd.read_csv(train_file)
val_All_data=pd.read_csv(val_file)
data_id=np.array(All_data['id'])
data_label=np.array(All_data['label'])
print(data_label)
val_data_label=np.array(val_All_data['label'])


#---------------weight and bias intialisation-------------------------------------------


train_data=All_data.iloc[:,1:785]
train_data=f_norm(train_data)
rows=train_data.shape[0]
col=train_data.shape[1]
print(rows)
print(col)
total_batch=int(rows/batch_size)

if pretrain=="false":
    W=initial_weight(col,num_hidden,sizes)
    b=initial_bias(col,num_hidden,sizes)

    b_len=len(b)
    for i_b in range(b_len):
        b_b_len=len(b[i_b])
        b_aa=np.array(b[i_b])
        b_aa.shape=(b_b_len,1)
        b[i_b]=b_aa

    for model_i in range(4):
        if model_i == 0 :
            opt= "adam"
            adam_train_obs=dict()
            adam_val_obs=dict()

        elif model_i == 1 :
            opt= "nag"
            nag_train_obs=dict()
            nag_val_obs=dict()
        elif model_i == 2 :
            opt= "momentum"
            momentum_train_obs=dict()
            momentum_val_obs=dict()
        else:
            opt= "gd"
            gd_train_obs=dict()
            gd_val_obs=dict()



    if opt=="momentum" or opt== "nag":
        previous_update_w=dict()
        previous_update_b=dict()
        for prv_uw in range(num_hidden+1):
            previous_update_w.update({prv_uw:np.zeros((W[prv_uw].shape[0],W[prv_uw].shape[1]))})
            previous_update_b.update({prv_uw:np.zeros((b[prv_uw].shape[0],b[prv_uw].shape[1]))})
    if opt== "nag":
        w_lookahead=dict()
        b_lookahead=dict()
        for prv_uw in range(num_hidden+1):
            w_lookahead.update({prv_uw:np.zeros((W[prv_uw].shape[0],W[prv_uw].shape[1]))})
            b_lookahead.update({prv_uw:np.zeros((b[prv_uw].shape[0],b[prv_uw].shape[1]))})


    if opt== "adam":
        m_w_adam=dict()
        v_w_adam=dict()
        m_b_adam=dict()
        v_b_adam=dict()
        for prv_uw in range(num_hidden+1):
            m_w_adam.update({prv_uw:np.zeros((W[prv_uw].shape[0],W[prv_uw].shape[1]))})
            v_w_adam.update({prv_uw:np.zeros((W[prv_uw].shape[0],W[prv_uw].shape[1]))})
            m_b_adam.update({prv_uw:np.zeros((b[prv_uw].shape[0],b[prv_uw].shape[1]))})
            v_b_adam.update({prv_uw:np.zeros((b[prv_uw].shape[0],b[prv_uw].shape[1]))})
        beta1=0.9
        beta2=0.999
        eps=1e-8




    print("total_batch : {}".format(total_batch))


    batch_var_count_adam=0

    W_b_parameters=dict()
    w_dictionary=dict()
    b_dictionary=dict()

    for iteration_var in range(max_iteration):
        print("iteration : {}".format(iteration_var))
        for batch_var in range(total_batch):

            processing_data=train_data.iloc[batch_var*batch_size:(batch_var*batch_size)+batch_size,0:784]
            #train_data=(train_data)/255
            rows=train_data.shape[0]

            #---------------------one-hot----------------------------------------------------------

            one_hot_count_flag=0
            for i in range(batch_size*batch_var,(batch_size*batch_var)+batch_size,1):
               # print("batch  value {} {}".format(batch_var,i))
                y_original=[0]*num_outputlayer
                y_original[data_label[i]]=1
                y_original=np.array(y_original)
                if one_hot_count_flag==0:
                    arr=y_original
                    one_hot_count_flag=1
                else:
                    arr= np.vstack([arr, y_original])


            if batch_size == 1:
                b_b_len=len(arr)
                #b_aa=np.array(b[i_b])
                arr.shape=(b_b_len,1)
               # b[i_b]=b_aa
            else:
                arr=np.transpose(arr)       
                #---------------forward propogation-------------------------------------------------
            W,b,pre_activation,activation=forward_propogation(W,b,processing_data,num_hidden,activation_name)


                #--------------back propogation-----------------------------------------------------

                #print(W[0])
            if opt == "gd":
                grad_weight,grad_bias=backward_propogation(W,b,arr,num_hidden,pre_activation,activation,processing_data,loss_name)
                W,b = update_paramters_gd(W,b,grad_weight,grad_bias,eta)
            if opt == "momentum":
                grad_weight,grad_bias=backward_propogation(W,b,arr,num_hidden,pre_activation,activation,processing_data,loss_name)
                W,b,previous_update_w,previous_update_b = update_paramters_momentum(num_hidden,W,b,grad_weight,grad_bias,eta,momentum,previous_update_w,previous_update_b)
            if opt == "nag":
                for nag_var in range(num_hidden+1):
                    w_lookahead[nag_var]=W[nag_var]+(momentum*previous_update_w[nag_var])
                    b_lookahead[nag_var]=b[nag_var]+(momentum*previous_update_b[nag_var])
                    grad_w,grad_b=backward_propogation(w_lookahead,b_lookahead,arr,num_hidden,pre_activation,activation,processing_data,loss_name)
                for nag_var2 in range(num_hidden+1):
                    previous_update_w[nag_var2] = (momentum*(previous_update_w[nag_var2])) + (eta*grad_w[nag_var2])
                    W[nag_var2] = W[nag_var2] - previous_update_w[nag_var2]                                 

                    previous_update_b[nag_var2] = (momentum*(previous_update_b[nag_var2])) + (eta*grad_b[nag_var2])
                    b[nag_var2] = b[nag_var2] - previous_update_b[nag_var2]

            if opt == "adam":
                grad_weight,grad_bias=backward_propogation(W,b,arr,num_hidden ,pre_activation,activation,processing_data,loss_name)
                for adam_var in range(num_hidden+1):
                    m_w_adam[adam_var]=(beta1*m_w_adam[adam_var]) + ((1-beta1)*grad_weight[adam_var])
                    v_w_adam[adam_var]=(beta2*v_w_adam[adam_var]) + ((1-beta2)*np.power((grad_weight[adam_var]),2))

                    m_w_adam[adam_var] = m_w_adam[adam_var]/(1-np.power(beta1,batch_var_count_adam+1))
                    v_w_adam[adam_var] = v_w_adam[adam_var]/(1-np.power(beta2,batch_var_count_adam+1))
                    W[adam_var] = W[adam_var] - (((eta*m_w_adam[adam_var])/(np.sqrt(v_w_adam[adam_var]+eps)))) 


                    m_b_adam[adam_var]=(beta1*m_b_adam[adam_var]) + ((1-beta1)*grad_bias[adam_var])
                    v_b_adam[adam_var]=(beta2*v_b_adam[adam_var]) + ((1-beta2)*np.power((grad_bias[adam_var]),2))

                    m_b_adam[adam_var] = m_b_adam[adam_var]/(1-np.power(beta1,batch_var_count_adam+1))
                    v_b_adam[adam_var] = v_b_adam[adam_var]/(1-np.power(beta2,batch_var_count_adam+1))

                    b[adam_var] = b[adam_var] - (((eta*m_b_adam[adam_var])/(np.sqrt(v_b_adam[adam_var]+eps)))) 
                batch_var_count_adam+=1


            if (batch_var%100)==99 :   
    #            dummy_label=np.array(All_data['label'])
                dummy_label=data_label
                train_data11=All_data.iloc[0:(batch_size*batch_var)+batch_size,1:785]
                dummy_label.reshape(len(dummy_label),1)
                dummy_label=dummy_label[0:(batch_size*batch_var)+batch_size]
                processing_data11=train_data11
                total_loss,accuracy,error=accuracy_error_calculation(W,b,num_hidden,processing_data11,dummy_label,loss_name,activation_name)      
                log_train_string+="Epoch %d, Step %d, Loss: %.2f, Error: %.2f, lr: %f \n" %(iteration_var,step_number,total_loss,error,eta)           

    #            dummy_label=np.array(val_All_data['label'])
                dummy_label=val_data_label
                train_data11=val_All_data.iloc[:,1:785]
                dummy_label=dummy_label[:]
                processing_data11=f_norm(train_data11)
                total_loss,accuracy,error=accuracy_error_calculation(W,b,num_hidden,processing_data11,dummy_label,loss_name,activation_name)
                log_val_string+="Epoch %d, Step %d, Loss: %.2f, Error: %.2f, lr: %f \n" %(iteration_var,step_number,total_loss,error,eta)           
                step_number+=1000 

            del arr
        if anneal == "true":
            print("previous_loss {}".format(anneal_previous_val_error))
            print("eta{}".format(eta))
            dummy_data=pd.read_csv('val.csv')
            dummy_label=np.array(dummy_data['label'])
            train_data11=dummy_data.iloc[:,1:785]
            dummy_label=dummy_label[:]
            processing_data11=f_norm(train_data11)
            total_loss,accuracy,error=accuracy_error_calculation(W,b,num_hidden,processing_data11,dummy_label,loss_name,activation_name)
            current_val_error = total_loss
            if current_val_error > anneal_previous_val_error:
                eta=eta/2
                iteration_var-=1
            anneal_previous_val_error=current_val_error


        w_dictionary.update({iteration_var+1:W})
        b_dictionary.update({iteration_var+1:b})






    #-----------------log files updation--------------
    parameters=( w_dictionary,b_dictionary)
    with open('test_submission_1.pickle', 'wb') as handle:
        pickle.dump(parameters, handle )

    print("printing data in train file")
    f_train_log.write(log_train_string)    
    print("printing data in val file")
    f_val_log.write(log_val_string)


    f_train_log.close()
    f_val_log.close()

    print("eta {}".format(eta))    


    train_data11=dummy_data.iloc[:,1:785]
    dummy_label=dummy_label[:]
    processing_data11=f_norm(train_data11)
    loss,accuracy,error=accuracy_error_calculation(W,b,num_hidden,processing_data11,dummy_label,loss_name,activation_name)
    print("accuracy {}".format(accuracy))

                    #-----------TEST_DATA---------------------------
    #-----------------------------------------------------------------------------------------------------

    dummy_data=pd.read_csv('test.csv')
    train_data=dummy_data.iloc[:,1:785]
    processing_data=f_norm(train_data)
    W,b,pre_activation,activation=forward_propogation(W,b,processing_data,num_hidden,activation_name)

    ap=activation[num_hidden]
    y_cap_len=ap.shape[1]

    pridict_label=dict()

    for y_cp_l in range(y_cap_len):
        asa=ap[:,y_cp_l]
        pridict_label.update({y_cp_l:np.argmax(asa)})

    f = open("useless_test.csv","w+")
    f.write("id,label\n")
    for label_num in range(len(pridict_label)):
        f.write(str(label_num)+ "," + str(pridict_label[label_num]) + "\n")
    f.close()
else:
   
	with open('test_submission_1.pickle', 'rb') as f:
	    W,b = pickle.load(f)
	activation_name="sigmoid"
	num_hidden=3
	dummy_data=pd.read_csv('test.csv')
	train_data=dummy_data.iloc[:,1:785]
	processing_data=f_norm(train_data)




	W,b,pre_activation,activation=forward_propogation(W,b,processing_data,num_hidden,activation_name)

	ap=activation[num_hidden]
	y_cap_len=ap.shape[1]

	pridict_label=dict()

	for y_cp_l in range(y_cap_len):
	    asa=ap[:,y_cp_l]
	    pridict_label.update({y_cp_l:np.argmax(asa)})
	    
	f = open("test_submission.csv","w+")
	f.write("id,label\n")
	for label_num in range(len(pridict_label)):
	    f.write(str(label_num)+ "," + str(pridict_label[label_num]) + "\n")
	f.close()


