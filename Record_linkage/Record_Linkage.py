import os
import collections
import random
import pandas as pd
import numpy as np
import copy
import re
import time
from scipy.sparse import csr_matrix
import sparse_dot_topn.sparse_dot_topn as ct
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

from sklearn.feature_extraction.text import TfidfVectorizer

golden_csv = pd.read_csv('SFDC_ext.csv', header=None)
sample_csv = pd.read_csv('EDI_Entry_0118.csv')

golden_csv.columns = [['name', 'mcn-id','existing-or-new','address1','address2','city','zip','country','temp1','temp2','temp3','temp4','temp5','website']]
golden_csv_cleaned = golden_csv.drop('temp1', axis=1)
golden_csv_cleaned = golden_csv_cleaned.drop('temp2', axis=1)
golden_csv_cleaned = golden_csv_cleaned.drop('temp3', axis=1)
golden_csv_cleaned = golden_csv_cleaned.drop('temp4', axis=1)
golden_csv_cleaned = golden_csv_cleaned.drop('temp5', axis=1)
golden_csv_cleaned['existing-or-new'] = golden_csv_cleaned['existing-or-new'].fillna('EC-Existing')
golden_csv_cleaned['city'] = golden_csv_cleaned['city'].fillna('TX')
golden_csv_cleaned['zip'] = golden_csv_cleaned['zip'].fillna('11111')
golden_csv_cleaned['website'] = golden_csv_cleaned['website'].fillna('www.google.com')
golden_csv_cleaned['address2'] = golden_csv_cleaned['address2'].fillna('temporary address')
golden_csv_cleaned['address1'] = golden_csv_cleaned['address1'].fillna('temporary address')
golden_csv_cleaned['country'] = golden_csv_cleaned['country'].fillna('US')

sample_csv_cleaned = sample_csv.drop('eu_address2', axis=1)
sample_csv_cleaned = sample_csv_cleaned.drop('eu_address3', axis=1)
sample_csv_cleaned = sample_csv_cleaned.drop('eu_address4', axis=1)
sample_csv_cleaned = sample_csv_cleaned.drop('eu_province', axis=1)
sample_csv_cleaned = sample_csv_cleaned.dropna(axis=0, how='any')

sample_csv_cleaned['name_address'] = sample_csv_cleaned['eu_name']+ ' ' + sample_csv_cleaned['eu_address1']+' '  + sample_csv_cleaned['eu_city']+ ' ' + \
sample_csv_cleaned['eu_zip_code']+ ' '  + sample_csv_cleaned['eu_state']+ ' ' + sample_csv_cleaned['eu_country']

a1 = golden_csv_cleaned['name'].values
a2 = golden_csv_cleaned['address1'].values
a3 = golden_csv_cleaned['address2'].values
a4 = golden_csv_cleaned['city'].values
a5 = golden_csv_cleaned['zip'].values
a6 = golden_csv_cleaned['country'].values

a = a1+' '+a2+' '+a3+' '+a4+' '+a5+' '+a6

train_arr = []
test_arr = []
mcn_arr_test = []
train_data_length=len(sample_csv_cleaned)
for i in range(0, int(train_data_length)):
    train_arr_temp = sample_csv_cleaned['name_address'].values[i]
    mcn_number_temp = sample_csv_cleaned['eu_mcn'].values[i]
    train_arr.append(train_arr_temp)
    mcn_arr_test.append(mcn_number_temp)


mcn_arr_golden = golden_csv_cleaned['mcn-id'].values.tolist()
train_arr = np.array(train_arr)

np.savetxt(r'test_data_big.txt', train_arr, fmt='%s', encoding='utf-8')
np.savetxt(r'golden_data_big.txt', a, fmt='%s', encoding='utf-8')

golden_data = a.tolist()
golden_flat_list = [item for sublist in golden_data for item in sublist]

test_data = train_arr.tolist()

test_dict = {}
for i in range(0, len(mcn_arr_test)):
    test_dict[mcn_arr_test[i]] = train_arr[i]
    
golden_dict = {}
# golden_dict.setdefault('dummy',[])
for i in range(0, len(mcn_arr_golden)):
    golden_dict[mcn_arr_golden[i][0]] = golden_flat_list[i]
    
test_data = test_data[0:1000]

import time
t1 = time.time()



choices = golden_dict.values()
score= 0
found_flag=0
not_found=[]
for i in range(0, len(test_data)-1):
    found_flag=0
    chlist=process.extract(test_data[i], choices, limit=2)
#     print("{},   --test--{}".format(test_data[i], mcn_arr_test[i]))
    
    for mcn,value in golden_dict.items():
        for j in range(0,2):
            if value == chlist[j][0]:
                if mcn_arr_test[i] == mcn:
                    score+=1
                    print('Done for test sample', i)
                    t = time.time()-t1
                    print("SELFTIMED:", t)
                    
#                     print("{},   --MDM--{}".format(chlist[j][0], mcn))
#                     print("===================================================")
                    found_flag=1
                    break
    if found_flag == 0:
        not_found.append(i)
     
print("Total matches found are {} out of {}".format(score, len(test_data)))        
print('accuracy is', 100* score/len(test_data), '%')
print('mismatched sample indexes in test data are:')
print(not_found)
print("===================================================")
print('We are done with 1st level of matching')

not_matched = []
for i in not_found:
    not_matched.append(train_arr[i])

not_matched_test  = copy.copy(not_matched)

test_df  = pd.DataFrame({'mcn':mcn_arr_test, 'name-address':train_arr})
golden_df = pd.DataFrame({'mcn':mcn_arr_golden, 'name-address':golden_flat_list})

names = pd.concat([test_df, golden_df])
len(names)

def get_matches_df(sparse_matrix, name_vector, top=1000):
    non_zeros = sparse_matrix.nonzero()
    
    sparserows = non_zeros[0]
    sparsecols = non_zeros[1]
    
    if top:
        nr_matches = top
    else:
        nr_matches = sparsecols.size
    
    left_side = np.empty([nr_matches], dtype=object)
    right_side = np.empty([nr_matches], dtype=object)
    similairity = np.zeros(nr_matches)
    
    for index in range(0, nr_matches):
        left_side[index] = name_vector[sparserows[index]]
        right_side[index] = name_vector[sparsecols[index]]
        similairity[index] = sparse_matrix.data[index]
    
    return pd.DataFrame({'left_side': left_side,
                          'right_side': right_side,
                           'similairity': similairity})
        

for i in range(0, len(golden_flat_list)):
    not_matched.append(golden_flat_list[i])

full_data = copy.copy(test_data)

for i in range(0, len(golden_flat_list)):
    full_data.append(golden_flat_list[i])




def ngrams_partial(string, n=1):
    string = re.sub(r'[,-./]|\sBD',r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]



company_names = full_data
vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams_partial)
tf_idf_matrix_partial = vectorizer.fit_transform(company_names)


def awesome_cossim_top(A, B, ntop, lower_bound=0):

    A = A.tocsr()
    B = B.tocsr()
    M, _ = A.shape
    _, N = B.shape
 
    idx_dtype = np.int32
 
    nnz_max = M*ntop
 
    indptr = np.zeros(M+1, dtype=idx_dtype)
    indices = np.zeros(nnz_max, dtype=idx_dtype)
    data = np.zeros(nnz_max, dtype=A.dtype)

    ct.sparse_dot_topn(
        M, N, np.asarray(A.indptr, dtype=idx_dtype),
        np.asarray(A.indices, dtype=idx_dtype),
        A.data,
        np.asarray(B.indptr, dtype=idx_dtype),
        np.asarray(B.indices, dtype=idx_dtype),
        B.data,
        ntop,
        lower_bound,
        indptr, indices, data)

    return csr_matrix((data,indices,indptr),shape=(M,N))

t1 = time.time()
matches_partial = awesome_cossim_top(tf_idf_matrix_partial, tf_idf_matrix_partial.transpose(), 40, 0.50)
t = time.time()-t1
print("SELFTIMED:", t)

matches_partial_df = get_matches_df(matches_partial, company_names,len(company_names))
matches_partial_df = matches_partial_df[matches_partial_df['similairity'] <= 0.99999] # Remove all exact matches

golden_partial_mcn_list = []
for i in range(0, len(matches_partial_df)):
    
    if matches_partial_df['right_side'].tolist()[i] in golden_dict.values():
        for mcn, value in golden_dict.items():
            if (value == matches_partial_df['right_side'].tolist()[i]):
                golden_partial_mcn_list.append(mcn)
                break
    else:
        golden_partial_mcn_list.append(0)
            
  
test_partial_mcn_list = []
for i in range(0, len(matches_partial_df)): 
    if matches_partial_df['left_side'].tolist()[i] in test_dict.values():   
        for mcn, value in test_dict.items():
            if (value == matches_partial_df['left_side'].tolist()[i]):
                test_partial_mcn_list.append(mcn) 
                break       
    else:
        test_partial_mcn_list.append(0)
            

score = 0
score_list=[]
matched_list_step2=[]
for i in range(0, len(test_partial_mcn_list)):
  
    if test_partial_mcn_list[i] == golden_partial_mcn_list[i]:
        if test_partial_mcn_list[i] != 0:
            score+=1
            # score_list is for removing dumplicates from the final matched data
            score_list.extend(([test_partial_mcn_list[i], matches_partial_df['left_side'].tolist()[i]]))
#             print("MCN Number in test data is {} and test sample is {}".format(test_partial_mcn_list[i], matches_partial_df['left_side'].tolist()[i]))
#             print("MCN Number in golden data is {} and golden sample is {}".format(golden_partial_mcn_list[i], matches_partial_df['right_side'].tolist()[i]))
#             print("score is {}".format(matches_partial_df['similairity'].tolist()[i]))
#             print('================================================================================') 
            matched_list_step2.append(matches_partial_df['left_side'].tolist()[i])
        if test_partial_mcn_list[i] == test_partial_mcn_list[i+1]:
            
            continue


            
accuracy = 0.5*len(set(score_list))/(len(not_matched_test)) *100 
                  
# print('====================================================')   
# print('====================================================')  
print("Accuracy on full data is {} %".format(accuracy))

matched_list_step2 = list(set(matched_list_step2))
print(matched_list_step2)

not_matched_test_step2 = [item for item in not_matched_test if item not in matched_list_step2]

test_partial_mcn_list_step2 = []
for i in range(0, len(not_matched_test_step2)):
    for mcn, value in test_dict.items():     
        if (value == not_matched_test_step2[i]):                            
            test_partial_mcn_list_step2.append(mcn)     
            break

  
test_data_split_features = []
for i in range(0, len(not_matched_test_step2)):
    z= not_matched_test_step2[i].split(' ')
    z[2:-2] = []
    z = ' '.join(z)
    test_data_split_features.append(z)


choices = golden_dict.values()
score= 0
found_flag=0
not_found={}
for k in 
for i in range(0, len(test_data_split_features)-1):
    found_flag=0
    not_found_iteration = 0
    chlist=process.extract(test_data_split_features[i], choices, limit=25)
    print("{},   --test--{}".format(test_data_split_features[i], test_partial_mcn_list_step2[i]))
    
    for mcn,value in golden_dict.items():
        for j in range(0,24):
            if value == chlist[j][0]:
                if test_partial_mcn_list_step2[i] == mcn:
                    score+=1
                    
                    print("{},   --MDM--{}".format(chlist[j][0], mcn))
                    print("===================================================")
                    found_flag=1
                    break
    if found_flag == 0:
        not_found_iteration.append(i)
        not_found[k] = not_found_iteration
     
print("Total matches found are {} out of {}".format(score, len(test_data_split_features)))        
print('accuracy is', 100* score/len(test_data_split_features), '%')
print('mismatched sample indexes in test data are:')
print(not_found)
print("===================================================")
print('We are done with 1st level of matching')



choices = golden_dict.values()
accuracy = []
not_found= {}
for k in range(25,40):
    not_found_iteration = []
    found_flag=0
    score= 0
    
    for i in range(0, len(test_data)-1):
        found_flag=0
        chlist=process.extract(test_data[i], choices, limit=k)
#         print("{},   --test--{}".format(test_data[i], mcn_arr_test[i]))

        for mcn,value in golden_dict.items():
            for j in range(0,k):
                if value == chlist[j][0]:
                    if mcn_arr_test[i] == mcn:
                        score+=1

#                         print("{},   --MDM--{}".format(chlist[j][0], mcn))
#                         print("===================================================")
                        found_flag=1
                        break
        if found_flag == 0:
            not_found_iteration.append(i)
            not_found[k] = not_found_iteration
    accuracy.append(100* score/len(test_data))
#     print("Total matches found are {} out of {}".format(score, len(test_data)))        
    print('accuracy is', 100* score/len(test_data), '%')
    print('mismatched sample indexes in test data are:')
#     print(not_found)
    
    print("===================================================")
    print('We are done with 1st level of matching')
print(accuracy)
print(not_found)