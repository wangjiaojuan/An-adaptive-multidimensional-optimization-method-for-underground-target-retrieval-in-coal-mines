import os
# 大文件夹的路径
root_dir = './Datasets/mydataset'
# 输出文本文件的名称
output_image_paths = root_dir.split('/')[2]+'_paths.txt'
# 遍历大文件夹，并写入图像路径到文本文件
with open(output_image_paths, 'w') as f:
    for category in os.listdir(root_dir):
        category_path = os.path.join(root_dir, category)
        # 确保是文件夹
        if os.path.isdir(category_path):
            for file in os.listdir(category_path):
                # 检查文件扩展名是否为图像格式
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif','.tif')):
                    file_path = os.path.join(category_path, file)
                    # 写入图像路径到文本文件
                    f.write(file_path + '\n')
print(f'所有图像路径已保存到 {output_image_paths}')

# 输出文本文件的名称
output_image_labels = root_dir.split('/')[2]+'_labels.txt'
label = 0
# 遍历大文件夹，并写入图像路径到文本文件
with open(output_image_labels, 'w') as f:
    for category in os.listdir(root_dir):
        category_path = os.path.join(root_dir, category)
        # 确保是文件夹
        if os.path.isdir(category_path):
            for file in os.listdir(category_path):
                # 检查文件扩展名是否为图像格式
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                    # 写入label到文本文件
                    f.write(str(label) + '\n')
        label = label +1
print(f'所有图像路径已保存到 {output_image_labels}')