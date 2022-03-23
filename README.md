**DocksInk Dashboard:** 
Select the data to be sent to AI from the web app of Predictive billing dashboard and select send data to AI. This trigger the lambda function which converts the data in JSON file and store in the S3 bucket 

**Lambda Function:** 
Once the data is uploaded to S3 bucket it triggers another lambda function which initiates the model training process in sagemaker

Below settings are used for lambda function “sagemakertrigger” to start the notebook instance

      Lambda>>functions>>create function
      Name : sagemakertrigger
      Role : sagemakertrigger-role-mxo45e3l
      Runtime : python 3.7  
      Select Create function.

Code for lambda function to initiate the sagemaker can be found in ***sagemakertrigger.py*** in ***predictive-ai - (Default)*** git repo.

**SageMaker:**

*Lifecycle script* (i.e. ***dev-config***) will first start a Jupyter notebook instance, install the dependencies setup environment variables then and run the model training script(***model_training.ipynb***).

In lifecycle script .env file is created in order to setup enviroment variables which will be removed post training job is completed.
***Lifecycle-config.sh*** in ***predictive-ai - (Default)*** git repository contains script for same.

dependencies installed using lifecycle script:

            pip install spacy
            python -m spacy download en_core_web_sm
            pip install openpyxl transformers sentence-transformers pandas
            pip install python-dotenv

After installing dependencies and setting up the enviroment variables it'll start execution of model training script(***model_training.ipynb***).

Data preprocessing happens in the Notebook. We convert the JSON data in the format which is accepted by the Model. A train and test dataset is created and saved to the S3 bucket(***dev-predictive-billing***).

An embedding file(***embeddings.sav***) is created using a pre-trained deep learning model(***SentenceTransformer('./efs/efs/models/bioclinicalbert')***) which was earlier trained on the medical and clinical data.  We leverage the knowledge of this model to create embedding. The time required to create embeddings is dependent on the number of data samples present since we create embeddings from scratch every time the process is initiated  and the time also depends on the type of instance we spinoff. If we use the GPU instance for jupyter notebook the embedding process is a bit quicker than the CPU instance.

An Elastic file system (***pythonml (fs-095543dfe911e19ee)***) is attached to the instance.
To create EFS:

      Amazon EFS>File systems>Create
      Name : pythonml (fs-095543dfe911e19ee)
      VPC : vpc-e59f389e( Main Dev NAT)
      Select Regional
      Select customise
      Select Next
	
Make sure that the mount target security group has an inbound rule That allows NFS access from Notebook instance security group  
      
      I.e. sg-60d71316 (Lambda Main Dev NAT)

To mount efs in notebook instance (***predictive-billing-ai***) we use below command in (***model_training.ipynb***):

      !sudo mount -t nfs \
          -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2 \
          {EFS_IP}:/ \
          ./efs

      !sudo chmod go+rw ./efs
For installing dependencies in efs we need to open terminal and create new directory in efs(./efs/efs/lib) and use requirements.txt for versions :

       	cd efs
		mkdir efs
		cd efs
		mkdir lib
		cd ../..
		pip install –target=/home/ec2-user/SageMaker/predictive-ai/efs/efs/lib -r requirements.txt

and to copy the embeddings file to the models folder in EFS we use below command in (***model_training.ipynb***):


      !sudo cp ./embeddings.sav ./efs/efs/models/embeddings.sav

Next a keyphrase extraction (Named Entity Recognition) model training process starts. Here it takes certain parameters like location from where to import data and location to save the output model, as well we provide the number of times the model should go through the complete training data (***Epochs***) it updates model weight after each epoch. To train this model a different GPU instance is started. GPU is used as it is a deep learning model.

We run training script on SageMaker by creating a Hugging Face Estimator. The Estimator handles end-to-end SageMaker training. There are several parameters defined in the Estimator:

entry_point specifies which fine-tuning script to use.(***run_ner.py***)
instance_type specifies an Amazon instance to launch.
Transformers and pytorch version.
hyperparameters specifies training hyperparameters. 
The following are hyperparameters passed:

                 'num_train_epochs': NUM_EPOCHS,
                 'per_device_train_batch_size': BATCH_SIZE,
                 'model_name_or_path': 'emilyalsentzer/Bio_ClinicalBERT',
                 's3_bucket': S3_BUCKET,
                 's3_folder': S3_TRAINING_FOLDER+'/data',
                 's3_model_folder': model_folder,
                 'output_dir': './models/',
                 'max_seq_length': 128,
                 'do_train': True,
                 'do_eval': True,
                 'overwrite_output_dir': True,
                 'save_strategy': "no"

The model training script downloads the data from S3 bucket(***dev-predictive-billing***), trains the model and saves the model in S3 bucket. For every model train run we save the model to S3 bucket.  This GPU instance is automatically terminated once the training job is completed. The model training time largely depends on the amount of data and number of epochs. For  the key extraction model to train properly at least 1000 samples to be provided.

Next we download the model from the s3 bucket check if the model accuracy is greater than 50% and more than that of previously trained model, if the accuracy is more than two conditions we copy the model to EFS.

	!sudo cp -r bert-clinical-ner ./efs/efs/models/
Once all the process is done we close the jupyter notebook instance.

**Lambda Function:**
Lambda function is triggered to performs the prediction job.
Lambda function needs to have below  settings:

	Name :pythonml
	Role : pythonml-role-pprl7ztu 
	Runtime : python 3.7

Resource base policy :

	Add permissions:
	Statement Id: python-mllambda-1
	Principal: arn:aws:iam::012959706571:user/PankajTamhane	
	Actions : lambda:InvokeFunction

Environment variables:

	Key: PYTHONPATH      Value:/mnt/efs/lib

VPC:

	Name : vpc-e59f389e (172.30.0.0/16) | Main Dev NAT
	Subnet : subnet-70bee72d (172.30.128.0/18) | us-east-1a, Main Dev NAT Private .128.0
	Security Group : sg-60d71316 (Lambda Main Dev NAT)

File System :

	File system Id : fs-095543dfe911e19ee 
	File system ARN :arn:aws:elasticfilesystem:us-east-1:012959706571:file-system/fs-095543dfe911e19ee
	
	Access point id :fsap-041630d041056d952 
	Access point ARN : arn:aws:elasticfilesystem:us-east-1:012959706571:access-point/fsap-041630d041056d952
S3 Trigger:

	Lambda>>Add Trigger
	Bucket : predictive-billing
	Prefix: medical/transcribe-jobs
	Suffix: .json

EFS is attached to the lambda function. We have the trained model artifacts in the EFS. We also copy the python dependencies in the EFS so that lambda obtains the dependencies from EFS. We use EFS to save on time required to download the model every time lambda is triggered. And it also provides quick model loading 
We load the embeddings file and NER model
First we extract the key phrases and run all the keyphrases through embedding and find the CPT code for each keyphrase.
Once all the CPT for phrases are identified the data is sent to the DI backend which then stores the data in the database and the mapped CPT data is available on UI
