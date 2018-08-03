
# coding: utf-8

# In[1]:


from __future__ import print_function

# $example on$
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier


from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# $example off$
from pyspark.sql import SparkSession


# In[2]:


if __name__ == "__main__":
    spark = SparkSession        .builder        .appName("RandomForestClassifierExample")        .getOrCreate()

    # $example on$
    # 加载并解析数据文件，将其转换为DataFrame
    data = spark.read.format("libsvm").load("./data/sample_libsvm_data.txt")


# In[3]:


# 索引标签，将元数据添加到标签列
# 适合整个数据集以包含索引中的所有标签
labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)

# 自动识别分类特征，并为它们编制索引
# 设置maxCategories，将具有> 4个不同值的要素视为连续
featureIndexer =    VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)


# In[4]:


# 训练一个RandomForest模型
rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=10)

# 将索引标签转换回原始标签
labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                               labels=labelIndexer.labels)


# In[5]:


# 管道中的链索引器和森林
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf, labelConverter])


# In[6]:


# 将数据分成训练和测试集（30％用于测试）
(trainingData, testData) = data.randomSplit([0.7, 0.3])


# In[7]:


# 训练模型，这也运行索引
model = pipeline.fit(trainingData)

# 作出预测
predictions = model.transform(testData)


# In[9]:


# 选择要显示的示例行
predictions.select("predictedLabel", "label", "features").show(5)


# In[10]:


# 选择（预测，真实标签）和计算测试误差
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))

rfModel = model.stages[2]
print(rfModel)  # summary only
# $example off$


# In[11]:


from pyspark.ml import PipelineModel
# 保存模型
modelPath = './model/random_forest_classifier_example_model'
model.write().overwrite().save(modelPath)


# In[12]:


spark.stop()

