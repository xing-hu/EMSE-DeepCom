# EMSE-DeepCom
The source code and dataset for EMSE-DeepCom

# Projects extracted from Github
The project information are listed in the file projects.txt. 
Each line represents a project which includes the GitHub username and project name connected by "_" 

# The distribution of the Java methods and classes in projects


# Data process
## Generate ASTs for Java methods
Command: `python3 get_ast.py source.code ast.json`
source.code:the source code file and each line represents one Java method.
ast.json: the ast file for Java method and each line represents one ast:

For Example:
```Java
public boolean doesNotHaveIds (){ 
  return getIds () == null || getIds ().getIds().isEmpty(); 
}
```
```
[
{"id": 0, "type": "MethodDeclaration", "children": [1, 2], "value": "doesNotHaveIds"}, 
    {"id": 1, "type": "BasicType", "value": "boolean"}, 
    {"id": 2, "type": "ReturnStatement", "children": [3], "value": "return"}, 
        {"id": 3, "type": "BinaryOperation", "children": [4, 7]}, 
            {"id": 4, "type": "BinaryOperation", "children": [5, 6]}, 
                {"id": 5, "type": "MethodInvocation", "value": "getIds"}, 
                {"id": 6, "type": "Literal", "value": "null"}, 
            {"id": 7, "type": "MethodInvocation", "children": [8, 9], "value": "getIds"}, 
                {"id": 8, "type": "MethodInvocation", "value": "."}, 
                {"id": 9, "type": "MethodInvocation", "value": "."}
 ]
```

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

