# Information Retrieval - BM25 Model


## What this programs can do

I have implemented both programs to deal with small and large corpus. They are in the separated ".py" files, which are both provided in this ".zip" archive. The python program "small.py" is used to deal with the small corpus, while "large.py" is used to deal with the large corpus. 

**The first time** running these programs, the files in the corresponding corpus would be read into a proper data structure as the Index, then it would be stored in to an external file. **Next time** running this, the index would not be created again, rather would be loaded from that existing external file, which is a json file. This external file is named as **"index.json"** in the same directory of the ".py" file you have executed.

Furthermore, in terms of the details in the indexing stage, the programs proceed the pre-process, which contains the processes like "extract the documents", "stopword removal", "stemming", "normalization" etc.

Users are able to determine whether run this program in **"manual"** model or **"evaluation"** mode.

When running in the **"manual"** mode, user can input queries in the command line, then presses the "Enter" key to execute this query. After that, the top 15 documents would be retrieved, which are sorted by bm25 similarity score in the descending order. The searching result has three columns: the rank, the document’s ID, and the similarity score.

The user can continue to be prompted to enter further queries until they type "QUIT". Moreover, "QUIT" command is not case sensitive.

When running in the **"evaluation"** mode, user do not need to input anything. Standard queries and relevance judgement will be loaded after index loading. Then these quires will be executed and contribute to the final score of evaluation. 7 different metrics will be used to evaluate this bm25 model, which are:

1. Precission
2. Recall
3. P@10
4. R-Precission
5. MAP
6. bpref
7. NDCG

After the evaluation process, the score of each of those metrics would be printed in the terminal. They are the average scores of all the standard queries.

Furthermore, if running in the **"evaluation"** mode, an output file would not only be created, but also be filed with required contents. This file is named as **"output.txt"** and will be created in the same directory of the ".py" file you have executed. The format of it meets the requirements, which have 6 fields on each line:  

1. The Query ID.
2. The string "Q0" (this is generally ignored)
3. The Document ID.
4. The rank of the document in the results for this query (starting at 1).
5. The similarity score for the document and this query.
6. The name of the run (my UCD student ID number).

My programs also record the time consumption of each process, including "creating index", "loading index", "evaluation" etc. These will all be shown in your terminal, then you can check them.





## How to run my programs

### For small corpus - "small.py"

1. First, you should copy the **"small.py"** file into your own "COMP3009J-corpus-small" directory, which contains the needed directories called "documents" and "files".

2. Then, as statements in "porter.py" said, you should copy the **"porter.py"** file into the same directory of "small.py".

3. You should let the position of your command line in this directory.

4. You can can run this program in 3 ways in the command line:

    1. "Manual" mode

        Input the following command in your command line:

        `python small.py -m manual`

        After you have been told that "Index loading finished". You will see a prompt of "Enter query:". You can enter a query here manually, then press the "Enter" key to execute searching. Then top 15 results will be printed in your terminal. After that you can continue to enter queries after the prompt of "Enter query:", or just enter a single command "QUIT" to stop this program. Moreover, "QUIT" command is not case sensitive.

    2. "Evaluation" mode

        Input the following command in your command line:

        `python small.py -m evaluation`

        After you have been told that "Index loading finished", the evaluation process would start. After you have been told "Evaluation finished", the evaluation results would be printed in your terminal, then you can check them.

    3. Without mode selection

        Input the following command in your command line:

        `python small.py`

        If no mode selected in the command line, the program will run in "manual" mode as default.


### For large corpus - "large.py"

1. First, you should copy the **"large.py"** file into your own "COMP3009J-corpus-large" directory, which contains the needed directories called "documents" and "files".

2. Then, as statements in **"porter.py"** said, you should copy the "porter.py" file into the same directory of "large.py".

3. You should let the position of your command line in this directory.

4. You can can run this program in 3 ways in the command line:

    1. "Manual" mode

        Input the following command in your command line:

        `python large.py -m manual`

        After you have been told that "Index loading finished". You will see a prompt of "Enter query:". You can enter a query here manually, then press the "Enter" key to execute searching. Then top 15 results will be printed in your terminal. After that you can continue to enter queries after the prompt of "Enter query:", or just enter a single command "QUIT" to stop this program. Moreover, "QUIT" command is not case sensitive.

    2. "Evaluation" mode

        Input the following command in your command line:

        `python large.py -m evaluation`

        After you have been told that "Index loading finished", the evaluation process would start. After you have been told "Evaluation finished", the evaluation results would be printed in your terminal, then you can check them.

    3. Without mode selection

        Input the following command in your command line:

        `python large.py`

        If no mode selected in the command line, the program will run in "manual" mode as default.
