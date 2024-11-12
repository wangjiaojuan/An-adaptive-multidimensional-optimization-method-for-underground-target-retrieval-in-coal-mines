from __future__ import division 
from C_feature_engineering_relation_function import*
import numpy as np
import time
import os
import warnings
warnings.filterwarnings("ignore")
def binary_datasets_feature(dir):
   for fi in os.listdir(dir):
      print(fi)
      if '.npy' in fi:
        try:
          if not os.path.exists('./binary_features/'+fi):
              datasets_feature = np.load('./features/'+fi)
              for i in range(datasets_feature.shape[0]):
                print(i)
                feature = datasets_feature[i]
                datasets_feature[i] = binary_feature(feature)
              datasets_feature=np.round(datasets_feature)
              datasets_feature = datasets_feature.astype(np.int8)
              np.save('./binary_features/'+fi,datasets_feature)
        except:
          print('load features error',fi)
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
    file_path = './results/'+Datasets_name+".txt"
    filewrite = open(file_path, "w")
    '''
    Result = ['all_feature_shuffleNetV2','all_feature_vgg16']
    '''
    Result = ['all_feature_shuffleNetV2','all_feature_vgg16','all_feature_resnet18','all_feature_densenet121',
              'all_feature_Alexnet','all_feature_convnext','all_feature_squeezenet',
              'all_feature_googlenet','all_feature_efficientnet','all_feature_mnasnet',
              'all_feature_regnet','all_feature_mobilenet']#'all_feature_HOG',,'all_feature_color','all_feature_lbp','all_feature_GIST','all_feature_Harris'
    
    for feature_name in Result:
        print("........................................comparision is",feature_name)
        print("........................................comparision is",feature_name,file=filewrite)
        # get feature
        origion_feature = np.load('./features/'+Datasets_name.split('_')[0]+'_feature_'+feature_name.split('_')[2]+'.npy')
        binary_feature = np.load('./binary_features/'+Datasets_name.split('_')[0]
                                 +'_feature_'+feature_name.split('_')[2]+'.npy')
        #origion_feature = np.load('./features/'+Datasets_name+'_feature_'+'all_feature_vgg16'.split('_')[2]+'.npy')
        #binary_feature = np.load('./binary_features/'+Datasets_name+'_feature_'+'all_feature_vgg16'.split('_')[2]+'.npy')
        # split retrieval image and database image
        re,database,origion_file = split_retrieval_datasets(Datasets_name)
        origion_ac=0
        binary_ac=0
        origion_time=0
        binary_time=0
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
            time1 = time.time()
            CNN_distance=euclidean_distance(origion_feature,path,database)
            result_CNN=query_result(CNN_distance,ac_label,database,Datasets_name,Top_N)
            time2 = time.time()
            color_distance=haming_distance(binary_feature,path,database)
            result_color=query_result(color_distance,ac_label,database,Datasets_name,Top_N)
            time3 = time.time()

            origion_time = origion_time + (time2 - time1)
            binary_time = binary_time + (time3-time2)
            origion_ac = origion_ac+result_CNN
            binary_ac = binary_ac+result_color

        origion_time = origion_time/len(re)
        binary_time = binary_time/len(re)
        origion_ac = origion_ac/len(re)
        binary_ac = binary_ac/len(re)

        print("origion_ac,binary_ac.................is",origion_ac,binary_ac)
        print("origion_time,binary_time.................is",origion_time,binary_time)

        print("origion_ac,binary_ac.................is",origion_ac,binary_ac,file=filewrite)
        print("origion_time,binary_time.................is",origion_time,binary_time,file=filewrite)

if __name__ == '__main__':
    for Datasets_name in ['mydataset']:#'CUMT-BelT','mydataset'
        for Top_N in [70]:
            print('Top_N',Top_N,Datasets_name)
            compare(Datasets_name,Top_N)

