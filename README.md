# Homework
打包文件夹可以从百度网盘下载  
链接：https://pan.baidu.com/s/1_ty1lzd9kIEkaet0LW1Y0g  
提取码：686A  
  
其中 test 和 train 是从 Kaggle 上直接下载下来的  
https://www.kaggle.com/competitions/dogs-vs-cats/data  
拉到最下面右下角下载DownloadAll  
  
第一种方法只需要run.py  
包含了预处理，模型搭建，训练及评价  
  
第二种方法要用到剩下所有的  
separate.py 可以把 train 里面的猫狗分开，另存为 train2  
test2 里面创建符号链接，这样不用再复制一遍，同时也满足了图片文件的读取要求  
predict.py 可以用来检查训练情况  
gap_get 和 gap_load 为导出并载入特征向量  
  
