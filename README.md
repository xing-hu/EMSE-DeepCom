# EMSE-DeepCom
The source code and dataset for EMSE-DeepCom

# Projects extracted from Github
These projects are listed in the file projects.txt. 
Each line represents a project which includes the GitHub username and project name connected by "_" 

# Evaluate Metrics
The evaluation scripts are listed in the file Scripts.
## The Sentence-level evaluation by NLTK:
Command: `python3 evaluation.py reference predictions`

## The Corpus-level evaluation by multi-bleu.perl:
Command: `perl multi-bleu.perl reference < predictions` 

## The METEOR evaluation by meteor 1.5:
Command: `java -Xmx2G -jar meteor-1.5.jar predictions reference -l en -norm`

reference: the ground-truth file (the test.token.nl file in our dataset).
predictions: the generated comments file.
Each line represents one sample.

