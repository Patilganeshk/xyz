## Model Training:

**EFS:**
1. Create an EFS with a certain VPC that will be utilised by Sagemaker and Lambda.
2. Make sure that the mount target security group has an inbound rule That allows NFS access from Notebook instance security group

**Notebook Instance**
1. Create notebook instance with same VPC as EFS
2. Make sure you've selected execution role 
3. Select security group same as EFS
4. You'll need to mention "predictive-ai" as default git repository for notebook instance

**Mount efs in Notebook and install dependencies**
1. open terminal once jupyter notebook is started 
2. make directory ./efs

			mkdir efs
3.run below command to mount efs in created directory and {EFS_IP} can be found in efs configuration settings or in Lifecycle-config.sh script in predictive-ai Repo.

	sudo mount -t nfs \
          -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2 \
          {EFS_IP}:/ \
          ./efs
4.For installing dependencies in efs we need create seprate directory in efs(./efs/efs/lib) and use requirements.txt for versions. Run following commands in same terminal:
	
	cd efs
	mkdir efs
	cd efs
	mkdir lib
	cd ../..
	pip install –target=/home/ec2-user/SageMaker/predictive-ai/efs/efs/lib -r requirements.txt
5.In order to create embeddings and save both ner model and sentence transformer models in efs run below commands:
	
	pip install -r requirements.txt
	python model_embeddings.py
	cp -R models /home/ec2-user/SageMaker/predictive-ai/efs/efs

**Lifeccycle configuration**
1. create lifecycle configuration
2. copy the script from ***Lifecycle-config.sh*** for seting up enviroment varibles, install dependencies and execute the ***Model_training.ipynb***.
3. make sure that u attach the lifecycle configuration to notebook instance we created.

**Lambda Function:**
1. Create a python3.7 lambda function and make sure the lambda has lambda execution and sagemaker trigger role
2. Add S3 trigger with bucket and file path of training json
3. Copy the ***sagemakertrigger.py*** code in lambda function 

This is setup that we need to train the model. Once the data is uploaded to S3 bucket it triggers lambda function which initiates the model training process in sagemaker. Lifecycle script will first start a Jupyter notebook instance, install the dependencies setup environment variables then and run the model training script.Data preprocessing happens in the Notebook. We convert the JSON data in the format which is accepted by the Model. A train and test dataset is created and saved to the S3 bucket. An embedding file is created using a pre-trained deep learning model which was earlier trained on the medical and clinical data. An Elastic file system is attached to the instance and we copy the embeddings file to the models folder in EFS. 

****
## Prediction:
1. create another lambda function to perform the prediction job.
2. make sure lambda function has lambda execution and sagemaker trigger role 
3. In resource base policy add permission for InvokeFunction
4. create enviroment variables like  PYTHONPATH  (Value:Value:/mnt/efs/lib) APIPATH(Value : Api path)
5. make sure you select the same vpc and security group as efs
6. Add S3 trigger with bucket and file path of transcription json
7. copy the script for prediction job from ***lambda.py*** from git repository.

Once a transcription json is uploaded to s3 bucket a Lambda function is triggered and performs the prediction job.EFS is attached to the lambda function. We have the trained model artifacts in the EFS. We also copy the python dependencies in the EFS so that lambda obtains the dependencies from EFS. We use EFS to save on time required to download the model every time lambda is triggered. And it also provides quick model loading. We load the embeddings file and NER model.First we extract the key phrases and run all the keyphrases through embedding and find the CPT code for each keyphrase.Once all the CPT for phrases are identified the data is sent to the DI backend which then stores the data in the database and the mapped CPT data is available on UI.
