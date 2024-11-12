from __future__ import division 
from C_feature_engineering_relation_function import*
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import random
warnings.filterwarnings("ignore")
import pandas as pd
from scipy import io

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
    random.shuffle(re)
    re = re[:50]
    return re,database,origion_file
def compare(Datasets_name):
    Result = ['all_feature_shuffleNetV2','all_feature_vgg16','all_feature_resnet18','all_feature_densenet121',
              'all_feature_Alexnet','all_feature_convnext','all_feature_squeezenet',
              'all_feature_googlenet','all_feature_efficientnet','all_feature_mnasnet',
              'all_feature_regnet','all_feature_mobilenet']#'all_feature_HOG',
    randomImg_image = np.zeros(shape=[50,len(Result)])
    feature_num = 0
    for feature_name in Result:
        print("........................................comparision is",feature_name)
        # get feature
        binary_feature = np.load('./binary_features/'+Datasets_name.split('_')[0]
                                 +'_feature_'+feature_name.split('_')[2]+'.npy')
        # split retrieval image and database image
        re,database,origion_file = split_retrieval_datasets(Datasets_name)
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
            binary_distance=haming_distance(binary_feature,path,database)
            result_binary=query_result(binary_distance,ac_label,database,Datasets_name)
            randomImg_image[i,feature_num] = result_binary
        feature_num = feature_num + 1
    print("origion_ac,binary_ac.................is",randomImg_image)
    return randomImg_image
def corr(Datasets_name,pandas_dataframe):
    pandas_dataframe = pd.DataFrame(pandas_dataframe)
    # 计算相关性系数矩阵
    corr_matrix = pandas_dataframe.corr()
    # 打印相关性系数矩阵
    print(corr_matrix)
    # 绘制热力图
    plt.figure(figsize=(24,12))
    x_ticks = ['shuffleNetV2','vgg16','resnet18','densenet121','Alexnet','convnext','squeezenet',
              'googlenet','efficientnet','mnasnet','regnet','mobilenet']
    y_ticks = ['shuffleNetV2','vgg16','resnet18','densenet121','Alexnet','convnext','squeezenet',
              'googlenet','efficientnet','mnasnet','regnet','mobilenet']
    ax = sns.heatmap(corr_matrix, cmap="YlGnBu",xticklabels=x_ticks, yticklabels=y_ticks, annot=True,fmt=".1g", linewidths=0.5,annot_kws={'size':16})
    plt.xticks(fontsize=16)
    plt.xticks(rotation=45)
    plt.yticks(fontsize=16)
    plt.yticks(rotation=0)
    plt.title('Correlation Matrix',y=-0.2,fontsize=16)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    #plt.show()
    #plt.savefig('./results/'+Datasets_name+".png", dpi=500,format="svg")
    plt.savefig('./results/'+Datasets_name+".jpg")


def corr_img():
    fig, axes = plt.subplots(2, 1, figsize=(24, 28))

    pandas_dataframe = np.load('./results/'+'CUMT-BelT'+'_randomImg.npy')
    pandas_dataframe = pd.DataFrame(pandas_dataframe)
    corr_matrix1 = pandas_dataframe.corr()

    pandas_dataframe = np.load('./results/'+'mydataset'+'_randomImg.npy')
    pandas_dataframe = pd.DataFrame(pandas_dataframe)
    corr_matrix2 = pandas_dataframe.corr()

    x_ticks = ['shuffleNetV2','vgg16','resnet18','densenet121','Alexnet','convnext','squeezenet',
              'googlenet','efficientnet','mnasnet','regnet','mobilenet']
    y_ticks = ['shuffleNetV2','vgg16','resnet18','densenet121','Alexnet','convnext','squeezenet',
              'googlenet','efficientnet','mnasnet','regnet','mobilenet']
    sns.heatmap(corr_matrix1,ax=axes[0], cmap="YlGnBu",xticklabels=x_ticks, yticklabels=y_ticks, annot=True,fmt=".1g", linewidths=0.5,annot_kws={'size':16})
    plt.xticks(fontsize=16)
    plt.xticks(rotation=45)
    plt.yticks(fontsize=16)
    cbar = axes[0].collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)     
    
    sns.heatmap(corr_matrix2,ax=axes[1], cmap="YlGnBu",xticklabels=x_ticks, yticklabels=y_ticks, annot=True,fmt=".1g", linewidths=0.5,annot_kws={'size':16})
    plt.xticks(fontsize=16)
    plt.xticks(rotation=45)
    plt.yticks(fontsize=16)
    cbar = axes[0].collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)    
    #plt.show()
    #plt.savefig('./results/'+Datasets_name+".png", dpi=500,format="svg")
    plt.savefig('./results/'+"7.jpg")

if __name__ == '__main__':
    '''
    corr_img()
    '''
    for Datasets_name in ['mydataset','CUMT-BelT']:#['CUMT-BelT','Holidays','UCMerced_LandUse','RS_images_2800','WANG']:#,'Holidays','UCMerced_LandUse','RS_images_2800','WANG']:
        randomImg_image = compare(Datasets_name)
        #np.save('./results/'+Datasets_name+'_randomImg.npy', randomImg_image)
        io.savemat('./results/'+Datasets_name+'_randomImg.mat',randomImg_image)
        
        #pandas_dataframe = np.load('./results/'+Datasets_name+'_randomImg.npy')
        #corr(Datasets_name,pandas_dataframe)
    