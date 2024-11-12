from __future__ import division 
from E_result_engineering_relation_function import*
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")

def split_retrieval_datasets(Datasets_name):
    # get all_image label
    filepath = Datasets_name+'_paths.txt'
    file = open(filepath)
    origion_file = file.readlines()
    file.close()
    origion_line = len(origion_file)
    # get retrieval image and database image
    if 'Holidays' in Datasets_name:  
        path=0
        re=[] 
        database=[]
        while path<origion_line:
            re.append(path)
            name_map=ac_img_label(origion_file)
            re_image = origion_file[path].split('\\')[1] #检索图像的类名
            start,end = name_map[re_image]
            num=end-start+1
            start=start+1
            while start<=end:
              database.append(start)
              start=start+1
            path=path+num
    else:
        path=0
        re=[] 
        database=[]
        while path<origion_line:
            re.append(path)
            database.append(path)
            path=path+1
    return re,database,origion_file
def compare(Datasets_name,Top_N):
    file_path = './results/'+Datasets_name+"_result.txt"
    filewrite = open(file_path, "w")

    print("........................................comparision is",Datasets_name)
    print("........................................comparision is",Datasets_name,file=filewrite)
    if Datasets_name == 'CUMT-BelT':
        print("result_1","result_2","result_3","weight_1","weight_2","weight_3",file=filewrite)
    if Datasets_name == 'mydataset':
        print("result_1","result_2","weight_1","weight_2",file=filewrite)
    # get feature
    if Datasets_name == 'CUMT-BelT':
        all_feature_1 = np.load('./binary_features/'+Datasets_name+'_feature_'+'all_feature_Alexnet'.split('_')[2]+'.npy')
        all_feature_2 = np.load('./binary_features/'+Datasets_name+'_feature_'+'all_feature_regnet'.split('_')[2]+'.npy')
        all_feature_3 = np.load('./binary_features/'+Datasets_name+'_feature_'+'all_feature_convnext'.split('_')[2]+'.npy')
    if Datasets_name == 'mydataset':
        all_feature_1 = np.load('./binary_features/'+Datasets_name+'_feature_'+'all_feature_shuffleNetV2'.split('_')[2]+'.npy')
        all_feature_2 = np.load('./binary_features/'+Datasets_name+'_feature_'+'all_feature_resnet18'.split('_')[2]+'.npy')
        # split retrieval image and database image
    re,database,origion_file = split_retrieval_datasets(Datasets_name)

    cal_ac=0
    neet_time=0
    #########################
    for i in range(len(re)):
        path=re[i]
        #获得本次检索的正确结果的索引
        name_map=ac_img_label(origion_file)
        re_image = origion_file[path].split('\\')[1]
        start,end = name_map[re_image]
        ac_label=[]
        start=start+1
        while start<=end:
            ac_label.append(start)
            start=start+1 
        ##retrieval
        if Datasets_name == 'CUMT-BelT':
            distance1=haming_distance(all_feature_1,path,database)
            result_1=query_result(distance1,ac_label,database,Datasets_name,Top_N)

            distance2=haming_distance(all_feature_2,path,database)
            result_2=query_result(distance2,ac_label,database,Datasets_name,Top_N)

            distance3=haming_distance(all_feature_3,path,database)
            result_3=query_result(distance3,ac_label,database,Datasets_name,Top_N)

            time1 = time.time()
            ours_ac,B = ours_query3(result_1,result_2,result_3,distance1,distance2,distance3,ac_label,database,Datasets_name,Top_N)
            time3 = time.time()

            print(result_1,result_2,result_3,ours_ac,B[0],B[1],B[2],file=filewrite)
            neet_time = neet_time + (time3 - time1)
            cal_ac = cal_ac+ours_ac
        if Datasets_name == 'mydataset':
            distance1=haming_distance(all_feature_1,path,database)
            result_1=query_result(distance1,ac_label,database,Datasets_name,Top_N)

            distance2=haming_distance(all_feature_2,path,database)
            result_2=query_result(distance2,ac_label,database,Datasets_name,Top_N)

            time1 = time.time()
            ours_ac,B = ours_query2(result_1,result_2,distance1,distance2,ac_label,database,Datasets_name,Top_N)
            time3 = time.time()    
            print(result_1,result_2,ours_ac,B[0],B[1],file=filewrite)
            neet_time = neet_time + (time3 - time1)
            cal_ac = cal_ac+ours_ac


    cal_ac = cal_ac/len(re)
    neet_time = neet_time/len(re)


    print("........................................comparision is")
    print("neet_time,cal_ac.................is",neet_time,cal_ac,file=filewrite)
    print("neet_time,cal_ac.................is",neet_time,cal_ac)


if __name__ == '__main__':
    for Datasets_name in ['mydataset','CUMT-BelT']:#['CUMT-BelT']:
        Top_N = 30
        for Top_N in [40,50,60,70,80,90,100]:
            Top_N = 10
            print('Top_N',Top_N)
            compare(Datasets_name,Top_N)

