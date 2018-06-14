
# coding: utf-8

# In[2]:


import pyspark.sql.types as typ

labels = [
    ('INFANT_ALIVE_AT_REPORT', typ.IntegerType()),
    ('BIRTH_PLACE', typ.StringType()),
    ('MOTHER_AGE_YEARS', typ.IntegerType()),
    ('FATHER_COMBINED_AGE', typ.IntegerType()),
    ('CIG_BEFORE', typ.IntegerType()),
    ('CIG_1_TRI', typ.IntegerType()),
    ('CIG_2_TRI', typ.IntegerType()),
    ('CIG_3_TRI', typ.IntegerType()),
    ('MOTHER_HEIGHT_IN', typ.IntegerType()),
    ('MOTHER_PRE_WEIGHT', typ.IntegerType()),
    ('MOTHER_DELIVERY_WEIGHT', typ.IntegerType()),
    ('MOTHER_WEIGHT_GAIN', typ.IntegerType()),
    ('DIABETES_PRE', typ.IntegerType()),
    ('DIABETES_GEST', typ.IntegerType()),
    ('HYP_TENS_PRE', typ.IntegerType()),
    ('HYP_TENS_GEST', typ.IntegerType()),
    ('PREV_BIRTH_PRETERM', typ.IntegerType())
]

schema = typ.StructType([
    typ.StructField(e[0], e[1], False) for e in labels
])

births = spark.read.csv('./data/births_transformed.csv.gz', 
                        header=True, 
                        schema=schema)


# In[3]:


import pyspark.ml.feature as ft

births = births     .withColumn(       'BIRTH_PLACE_INT', 
                births['BIRTH_PLACE'] \
                    .cast(typ.IntegerType()))


# In[4]:


encoder = ft.OneHotEncoder(
    inputCol='BIRTH_PLACE_INT', 
    outputCol='BIRTH_PLACE_VEC')


# In[5]:


featuresCreator = ft.VectorAssembler(
    inputCols=[
        col[0] 
        for col 
        in labels[2:]] + \
    [encoder.getOutputCol()], 
    outputCol='features'
)


# In[6]:


import pyspark.ml.classification as cl


# In[7]:


logistic = cl.LogisticRegression(
    maxIter=10, 
    regParam=0.01, 
    labelCol='INFANT_ALIVE_AT_REPORT')


# In[8]:


from pyspark.ml import Pipeline

pipeline = Pipeline(stages=[
        encoder, 
        featuresCreator, 
        logistic
    ])


# In[9]:


births_train, births_test = births     .randomSplit([0.7, 0.3], seed=666)


# In[10]:


model = pipeline.fit(births_train)
test_model = model.transform(births_test)


# In[12]:


test_model.take(1)


# In[13]:


import pyspark.ml.evaluation as ev

evaluator = ev.BinaryClassificationEvaluator(
    rawPredictionCol='probability', 
    labelCol='INFANT_ALIVE_AT_REPORT')

print(evaluator.evaluate(test_model, 
     {evaluator.metricName: 'areaUnderROC'}))
print(evaluator.evaluate(test_model, {evaluator.metricName: 'areaUnderPR'}))


# In[14]:


pipelinePath = './model/infant_oneHotEncoder_Logistic_Pipeline'
pipeline.write().overwrite().save(pipelinePath)


# In[15]:


loadedPipeline = Pipeline.load(pipelinePath)
loadedPipeline     .fit(births_train)    .transform(births_test)    .take(1)


# In[16]:


from pyspark.ml import PipelineModel

modelPath = './model/infant_oneHotEncoder_Logistic_PipelineModel'
model.write().overwrite().save(modelPath)

loadedPipelineModel = PipelineModel.load(modelPath)
test_loadedModel = loadedPipelineModel.transform(births_test)

