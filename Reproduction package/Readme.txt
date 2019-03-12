###########################################################################################################################################################

This is the Readme file for the reproduction package of the paper: "Automated Software Vulnerability Assessment with Concept Drift", accepted for publication at Mining Software Repositories 2019.

###########################################################################################################################################################

The package contains the following folders and files:
1. Data: containing all crawled NVD data including the raw descriptions, preprocessed descriptions and seven vulnerability characteristics of CVSS 2 we used in our work.

2. Source code files
	2.1. RQ1-2_Time-Validation.py: implementation of our time-based k-fold cross-validation described in section III.C. This module is used to provide answers for both sections V.A and V.B.
	
	2.2. RQ1_Non-Temporal-Validation.py: implementation of a non-temporal stratified cross-validation. This module is used to provide the quantitative comparison between our time-based cross-validation and a non-temporal cross-validation as described in section V.A.
	
	2.3. RQ3_CWM_Testing.py: implementation of our character-word model to handle concept drift for vulnerability assessment as described in III.A. This module is also used to provide results for comparison with the other two baselines: Word-only and Character-only models in section V.C.
	
	2.4. RQ3_WoM_Testing.py: implementation of word-only models (without handling concept drift). This is the first baseline model for comparison with our proposed character-word model as given in section V.C.
	
	2.5. RQ3_CoM_Testing.py: implementation of character-only models. This is second baseline model for comparison with our proposed character-word model as given in section V.C.
	
	2.6. RQ4-LSA_Testing.py: implementation of Latent Semantic Analysis described in section V.D.
	
	2.7. RQ4-fastText_Testing.py: implementation of fastText model trained on NVD descriptions described in section V.D.
	
	2.8. RQ4-fastText_Wiki_Testing.py: implementation of fastText model trained on Wikipedia English pages described in section V.D. It is noted that since the pre-trained fastText embeddings are very heavy (~6GB), we do not include them in package. However, it can be downloaded from https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md. and then saved as "crawl-300d-2M-subword.bin" in the same folder to run this module.
	
	2.9. CWM_Model_Saving.py: saving the character and word feature models as well as classification models for future vulnerability assessment (i.e., prediction of seven vulnerability characteristics).
	
	2.10. VCs_Prediction.py: prediction module of seven vulnerability characteristics with new NVD description as input.
	
3. Models: containing the trained feature and classification models based on the optimal NLP representations and classifiers, respectively, identified in the model selection process described in section III.C.

All the source codes are written in Python. In order to run them, the following external libraries are required:
1. numpy
2. pandas
3. scipy
4. scikit-learn
5. pickle
6. nltk
7. gensim
8. xgboost
9. lightgbm
