import time
import json
import os
import math
import sys
import getopt
import re

import porter


''' constants '''
# directories
DOCUMENT_DIR = 'documents/'
FILE_DIR = 'files/'
INDEX_FILE_NAME = 'index.json'
STOPWORDS_FILE_NAME = 'stopwords.txt'
STANDARD_QUERIES_FILE_NAME = 'queries.txt'
RELEVANCE_JUDGMENTS_FILE_NAME = 'qrels.txt'
OUTPUT_FILE_NAME = 'output.txt'
# other constants
UCD_STUDENT_NUMBER = '19206218'


''' other global variables '''
# --- for storing data ---
stopwords = set()
normalize_cache = dict()            # {'before normalizing': 'after normalizing'} 
stem_cache = dict()                 # {'before stemming': 'after stemming'} 
index_dic = dict()                  # {docID: {term: count}}
stemmer = porter.PorterStemmer()    # an instance of poterStemmer class
# --- for BM25 ---
N = 0               # total number of documents in the collection
avg_doclen = 0      # average length of documents in the collection
k = 1
b = 0.75
ni_dict = dict()    # records the number of doc in whole collection that contain each term. {term: number of docs} 
lengths = dict()    # records the length of each doc. {docID: length}
# --- for Evaluation ---
standard_queries = dict()   # {query ID: query str}
relevance_dict = dict()     # {queryID: {docID: relevance_score}}


''' ----------------------------------------------- Loading Functions ----------------------------------------------- '''   

def load_stopwords():
    """
        Read stopwords from the file 'stopwords.txt' into the stopwords set
    """
    # filename of stopwords.txt
    file_name = FILE_DIR + STOPWORDS_FILE_NAME
    # read stopwords    
    with open(file_name, 'r') as f:
        global stopwords
        stopwords = set(f.read().split())


def normalize(token:str):
    """
        This function will remove the punctuations at the beginning and end of the token.
        Then the punctuations inside the token will be removed as well.
    """
    # punctuations at both end
    punctuation = ''' /!`~"',;:.-?)([]<>*#_@+-=|$%^&\n\t\r'''
    result = token.strip(punctuation)
    # inner punctuations
    if result != "":
        result = result.replace('.', '').replace('-', '')
    return result


def stemming(term: str):
    """
        This function turns a term into stem by using the 'porter.py' library.
        Stemming is quite a time consuming procedure, so we optimize this by using Cache.
        For example, if a word in file no.10 has once been stemmed in file no.1, 
        we do not call stem function for it again, 
        rather we get the result from cache, which saves us a lot of time.

        :param term: A String of term
        :returns: A string of stemmed term (stem)
    """
    # only stem the terms that have not been stemmed before
    if term not in stem_cache:
        # add to cache
        stem_cache[term] = stemmer.stem(term)

    # get from cache
    term = stem_cache[term]

    return term


def store_index():
    """
        This function writes the index into a ".json" file
        The json file stores a list, which contains 5 dictionaries:
            index_dic, ni_dict, lengths, N (number of docs) and avg_doclen
        Consequently, we do not calculate them every time run the program.
    """
    # write the index into json file
    with open(INDEX_FILE_NAME, 'w') as f:
        lst = [index_dic, ni_dict, lengths, N, avg_doclen]
        json.dump(lst, f, indent=4, separators=(',', ': '))


def create_index():
    """
        Create the index, which records "which term appears how many times in which doc"
        {docID: {term: freq}}

        This functions assembles other subfunctions:
        The doc reading, term pre-processing and index generating.

        The first time running this program, this function should be called.
    """
    # the total length of the whole collection (for calculating the average doc len)
    sum_doc_len = 0

    # loop through all the files and subdirectories in the document root directory
    # (each folder)
    for root, dirnames, doc_ids in os.walk(DOCUMENT_DIR):
        # each doc in this folder
        for doc_id in doc_ids:
            if not doc_id.startswith('.'):  # avoid ".DS_store" file
                
                # update the total number of documents in collection (N)
                global N
                N += 1

                # term dict for this doc {term: freq}
                term_dict = dict()

                # the length of this document (how many terms)
                doc_len = 0

                # concatenate to get the whole complete name
                doc_name = os.path.join(root, doc_id)
                # read this document
                with open(doc_name, 'r', encoding='utf-8') as f:

                    # all_words = f.read().lower().split()

                    all_words = re.split(r'[`_=~!@#$%^&*()+\[\]{};\\:"|<,/<>?\s*]', f.read().lower())        # GOOD


                    # preprocess
                    for word in all_words:
                        # normalize
                        if word not in normalize_cache:
                            normalize_cache[word] = normalize(word)
                        term = normalize_cache[word]

                        # stopword removal
                        if term != "" and term not in stopwords:

                            # increase the doc length by 1
                            doc_len += 1

                            # stemming (use cache)
                            term = stemming(term)

                            # not first time find this term in this doc
                            if term in term_dict:
                                # update frequency in this doc
                                term_dict[term] += 1

                            # the first time
                            else:
                                # update frequency in this doc
                                term_dict[term] = 1
                                # count this file as containing this term
                                if term in ni_dict:
                                    ni_dict[term] += 1
                                else:
                                    ni_dict[term] = 1

                # record doc length
                lengths[doc_id] = doc_len
                # add this doc length to the sum (for calculating the average)
                sum_doc_len += doc_len

                # record index
                index_dic[doc_id] = term_dict

    # calculate the average doc length
    global avg_doclen
    avg_doclen = sum_doc_len / N

    # after creating the whole index dictionary in memory
    # we store the index info into an external file
    store_index()


def load_index():
    """
        This functions determines whether we can load the index info from
        existing 'index.json' file or we have to create one.

        The first time we run this script, this function will choose to 
        read in documents and generate an external index file. 

        In terms of other time, index info would be loaded from existing 'index.json' file.
    """
    # Check is this the first time running this program
    # by checking whether the external index file exists
    is_index_exist = os.path.exists(INDEX_FILE_NAME)

    # not the first time, we load the index from external file
    if is_index_exist:
        with open(INDEX_FILE_NAME, 'r') as f:
            # load the whole list from json file, which contains 3 dictionaries
            lst = json.load(f)
            # initialize the index_dict
            global index_dic
            index_dic = lst[0]
            # initialize the ni_dict
            global ni_dict
            ni_dict = lst[1]
            # initialize the lengths
            global lengths
            lengths = lst[2]
            # initialize the total document number
            global N
            N = lst[3]
            # initialize the average document length
            global avg_doclen
            avg_doclen = lst[4]

    # if this is the first time, we have to read in documents and create an index
    else:
        create_index()


''' ----------------------------------------------- Query Handling Functions ----------------------------------------------- '''   


def calculate_bm25_sim(query_terms: list, term_dict: dict, doc_id: str) -> float:
    """
        This function calculates the bm25 similarity between the given query and document.

        :param query_terms: A list of terms represent the query
        :param term_dict: The term dictionary of the document
        :param doc_id: The id of the document
        :returns: A float number represents the bm25 similarity between the given query and document.
    """
    bm25_sim = 0

    # loop through the query terms
    for term in query_terms:
        # check if the term in both the query and this document
        # we add the compute result of this term into the bm25 similarity of this document
        if term in term_dict:
            # prepare the parameters for calculating
            n = ni_dict[term]                           # number of documents in the collection that contain this term
            freq = term_dict[term]                      # the frequency of this term in this document
            doc_len = lengths[doc_id]                   # the length of this document
            
            # calculate how much this term contributes to the bm25 similarity
            left_part = (freq * (1+k)) / (freq + k*((1-b) + ((b*doc_len) / avg_doclen)))
            right_part = math.log2((N - n + 0.5) / (n + 0.5))

            bm25_sim += left_part * right_part
    
    return bm25_sim


def generate_bm25_dict(query_terms: list):
    """
        Calculate the bm25 similarity between the given query and every document.
        This will be stored in a dictionary {key: doc_id, value: bm25 similarity}

        :param query_terms: A list of terms represent the query
        :returns: A dict records the bm25 similarity between the given query and every document
    """
    sim_dict = dict()

    # loop through the whole collection of documents
    for doc_id, term_dict in index_dic.items():

        # calculate the bm25 similarity
        bm25_sim = calculate_bm25_sim(query_terms, term_dict, doc_id)

        # write the similarity into the dictionary
        sim_dict[doc_id] = bm25_sim
    
    return sim_dict


def print_query_result(query: str, sorted_doc_ids: list, sim_dict: dict):
    """
        This function prints out the query result, which is 
        a list of 15 most relevant documents, according to the BM25 IR Model, 
        sorted beginning with the highest similarity score.
        The output contains three columns: 
            1. the rank, 
            2. the document ID, 
            3. the bm25 similarity score
        
        :param query: The query string
        :param sorted_doc_ids: A list of document IDs sorted by bm25 similarity
        :param sim_dict: A dictionary stores the bm25 similarity of each document
    """
    print("Results for query [{}]:".format(query))
    
    for index, doc_id in enumerate(sorted_doc_ids):
        print("{} {} {:.6f}".format(index + 1, doc_id, sim_dict[doc_id]))

        if index == 14:
            break


def determine_ret_count(sim_dict: dict, sorted_doc_ids: list):
    """
        Determine how many document should be returned as relevant in a search of a query.

        :param sim_dict: A dictionary stores the similarity score between this query and every document.
        :param sorted_doc_ids: A list of ids of retrieved documents (all), sorted by relevance. 
        :returns: An appropriate number (int) of documents should be returned as retrieved
    """
    # the difference between the highest score and the lowest score
    diff = sim_dict[sorted_doc_ids[0]] - sim_dict[sorted_doc_ids[-1]] 

    # the bottom limit of the similarity score, that a document can be retrieved as relevant. 
    # This can balance the score of "precision" and "recall"
    # (how many percent of the similarity score interval would be treated as relevant)
    threshold = sim_dict[sorted_doc_ids[0]] - (diff * (47/100))   
    
    # determine the retrieved docs according to the threshold
    count_relevance = 0
    for doc_id in sorted_doc_ids:
        if sim_dict[doc_id] >= threshold:
            count_relevance += 1
        else:
            break

    # at least one returned document
    if count_relevance == 0:
        count_relevance = 1

    # for test
    # return 15
    return count_relevance


def query_processing(query: str):
    """
        Turn a query string into a list of terms
    """
    query_terms = []
    query_words = re.split(r'[`_=~!@#$%^&*()+\[\]{};\\:"|<,/<>?\s*]', query.lower())

    for word in query_words:
        # normalize
        if word not in normalize_cache:
            normalize_cache[word] = normalize(word)
        term = normalize_cache[word]

        # stopword removal
        if term != "" and term not in stopwords:
            # stemming
            term = stemming(term)
            # add to result
            query_terms.append(term)

    return query_terms


def search(query: str, mode: str):
    """
        This function searches a single query by calculating the bm25 similarity.
        An appropriate number of documents will be retrieved as the result.
        If the mode of this search is "manual", 15 most relevant document id will be returned.
        If the mode of this search is "evaluation", the number of retrieved documents will be determined (do a trade-off on "precision" and "recall")

        :param query: A string represents a query sentence
        :param mode: The model of this searching, can be "manual" or "evaluation" 
        :returns: A tuple contains two elements. 
                    1. A dictionary stores the similarity score between this query and every document. 
                    2. A list of retrieved document ids, which are sorted by relevance.
    """
    # do pre-process on the query string, make it into a list of terms
    query_terms = query_processing(query)

    # calculate the bm25 similarity between this query and every document
    # generate a data structure of dict to record this. (key: doc_id, value: bm25 similarity)
    sim_dict = generate_bm25_dict(query_terms)

    # sort the dict by similarity
    sorted_doc_ids = sorted(sim_dict, key=sim_dict.get, reverse=True)

    # determine how many result should be retrieved (trade-off on "precision" and "recall")
    if mode == "manual":
        ret_num = 15

    elif mode == "evaluation":
        ret_num = determine_ret_count(sim_dict, sorted_doc_ids)

    else:
        # wrong mode selection
        sys.exit(1)

    # slice the sorted id list, remain only the id we regard as "retrieved"
    sorted_doc_ids = sorted_doc_ids[:ret_num]

    return sim_dict, sorted_doc_ids


def listen_query():
    """
        This function performs a infinite loop to listen to the user input.
    """
    while True:

        # get the user query
        query = input("Enter query: ")

        # check quit command
        if query.lower() == 'quit':
            sys.exit(0)

        # search this query
        sim_dict, sorted_doc_ids = search(query=query, mode='manual')

        # print out the result
        print_query_result(query, sorted_doc_ids, sim_dict)


''' ----------------------------------------------- Evaluation Part ----------------------------------------------- '''

def load_queries():
    """
        Load the standard queries from "queries.txt" into memory.
    """
    # concatenate to get the filename of standard queries
    filename = FILE_DIR + STANDARD_QUERIES_FILE_NAME
    # read the file
    with open(filename, 'r') as f:
        # each line is a single query started with its ID
        for line in f:
            # strip off the '\n' on this line
            line = line.strip()
            if line:
                # divide the query id and query content
                divider = line.find(" ")
                query_id = line[:divider]
                query = line[divider + 1:]
                # record this query into the global query dict
                standard_queries[query_id] = query


def load_relevance_judgments():
    """
        This function loads relevance judgments from "qrels.txt" into the memory.
    """
    # concatenate to get the filename of relevance judgments
    filename = FILE_DIR + RELEVANCE_JUDGMENTS_FILE_NAME
    # read file
    with open(filename, 'r') as f:

        # each line records the relevance between a query and a document
        for line in f:
            judgment_info = line.split()    # [queryID, 0, docID, relevance_score]

            # turn relevance_score from str to int
            relevance_score = int(judgment_info[3])

            # write into the memory
            if judgment_info[0] not in relevance_dict:
                # create a new key in the outer dict with an initial inner dict
                relevance_dict[judgment_info[0]] = {judgment_info[2]: relevance_score}
            else:
                # add a new k-v pair into the inner dict of this query
                relevance_dict[judgment_info[0]][judgment_info[2]] = relevance_score        


def get_DCG_vector(gain_vector: list):
    """
        This is a tool provided for the function "calculate_NDCG_at_n".
        This function calculate the Discounted Cumulated Gain (DCG) vector for a given Gain (G) vector.

        :param gain_vector: A gain vector of a query result
        :returns: A Discounted Cumulated Gain (DCG) vector
    """
    # Discounted Cumulated Gain vector (first one is always the same value as the first one in gain vector)
    DCG_vector = [gain_vector[0]]

    # loop through the gain vector from rank2
    for index, g in enumerate(gain_vector[1:]):
        # if G == 0, the DCG should be same as the last one
        if g == 0:
            DCG_vector.append(DCG_vector[-1])
        else:
            # calculate the dcg (Discounted Cumulated Gain) at this rank
            rank = index + 2
            dcg = (g / math.log2(rank)) + DCG_vector[-1]
            DCG_vector.append(dcg)

    return DCG_vector


def calculate_NDCG_at_n(sorted_doc_ids: list, judgment_dict: dict, n: int):
    """
        This function calculates the "NDCG" vector of a given query.
        Then return the n th dimension in this vector as the score of "NDCG@n".

        :param sorted_doc_ids: A list of ids of retrieved documents, sorted by relevance. 
        :param judgment_dict: The judgment dictionary of this query, which is the inner dict of "relevance_dict" and contains ideal result documents and their relevant score. {docID, score}
        :param n: The n of "NDCG@n". (which rank in the NDCG vector)
        :returns: The "NDCG@n" score of bm25 model based this single query. 
    """
    # sort the judgment_dict in and descending order (4, 3, 2, 1)
    sorted_judgment_keys = sorted(judgment_dict, key=judgment_dict.get, reverse=True)

    # -- calculate the gain (G) vector
    gain_vector = []
    # loop through the retrieved documents from top to bottom
    for doc_id in sorted_doc_ids[:n]:
        # relevant
        if doc_id in judgment_dict:
            gain_vector.append(judgment_dict[doc_id])
        # non-relevant
        else:
            gain_vector.append(0)

    # -- calculate the ideal gain (IG) vector
    IG_vector = []
    gain_vector_len = len(gain_vector)

    for doc_id in sorted_judgment_keys[:gain_vector_len]:
        # get document id in the order from high judgement score to low
        IG_vector.append(judgment_dict[doc_id])

    # insert 0 scores at the end to make the length same as the retrieved list
    IG_vector_len = len(IG_vector)
    if IG_vector_len < gain_vector_len:
        for i in range(gain_vector_len - IG_vector_len):
            IG_vector.append(0)  

    # -- get Discounted Cumulated Gain (DCG) vector
    DCG_vector = get_DCG_vector(gain_vector)

    # -- get ideal Discounted Cumulated Gain (IDCG) vector
    IDCG_vector = get_DCG_vector(IG_vector)

    # -- normalize the DCG vector by dividing the IDCG vector, then we get Normalized Discounted Cumulated Gain (NDCG) vector
    # here we only need to know the last score in the NDCG vector, this is NDCG@n
    return DCG_vector[-1] / IDCG_vector[-1]


def begin_evaluation():
    """
        This function loop through all the standard queries and execute searching on each of them.
        An output.txt file will be generated, and the result of each query will be written in it.
        The bm25 model will be evaluated on different metrics, and the average score of it in each metric will be printed out.
    """
    # initialize the scores of each metric
    sum_precision = 0
    sum_recall = 0
    sum_p_at10 = 0
    sum_R_precision = 0
    sum_MAP = 0
    sum_bpref = 0
    sum_NDCG = 0

    # initialize the output.txt file
    if os.path.exists(OUTPUT_FILE_NAME) and os.path.isfile(OUTPUT_FILE_NAME):
        # delete this file, we will generate a new one latter.
        os.remove(OUTPUT_FILE_NAME)

    # open the output file for writing the result into the "output.txt" file
    with open(OUTPUT_FILE_NAME, 'a') as output_file:

        # loop through all the standard queries to execute each of them
        for query_id, query in standard_queries.items():
            # search this query
            sim_dict, sorted_doc_ids = search(query=query, mode='evaluation')

            # get the judgment dictionary of this query,
            # this contains ideal result documents and their relevant score. {docID, score}
            judgment_dict = relevance_dict[query_id]

            # count the number of judged relevant documents of this query
            rel_count = len(judgment_dict)


            """ test optimize (computing of each metric share a single loop) """
            rel_ret_count = 0
            rel_ret_count_at_10 = 0
            rel_ret_count_at_R = 0
            recalled_point_count = 0
            sum_precision_at_recalls = 0
            non_rel_count = 0
            sum_bpref_for_this = 0

            for index, doc_id in enumerate(sorted_doc_ids):
                # relevant
                if doc_id in judgment_dict :
                    # for precision and recall
                    rel_ret_count += 1
                    
                    # for p@10
                    if index + 1 <= 10:
                        rel_ret_count_at_10 += 1

                    # for R-precision (p@R)
                    if index + 1 <= rel_count:
                        rel_ret_count_at_R += 1

                    # for AP and MAP
                    recalled_point_count += 1
                    if recalled_point_count <= rel_count:
                        sum_precision_at_recalls += recalled_point_count / (index + 1)

                    # for bpref
                    if non_rel_count < rel_count:
                        sum_bpref_for_this += 1 - (non_rel_count / rel_count)
                    
                # non-relevant
                else:
                    # for bpref
                    non_rel_count += 1

            # calculate scores of this query
            precision = rel_ret_count / len(sorted_doc_ids)
            recall = rel_ret_count / rel_count
            p_at_10 = rel_ret_count_at_10 / 10
            r_precision = rel_ret_count_at_R / rel_count
            ap = sum_precision_at_recalls / rel_count
            bpref = sum_bpref_for_this / rel_count

            # add score of this query to the sum
            sum_precision += precision
            sum_recall += recall
            sum_p_at10 += p_at_10
            sum_R_precision += r_precision
            sum_MAP += ap
            sum_bpref += bpref
            
            # calculate NDCG@10
            sum_NDCG += calculate_NDCG_at_n(sorted_doc_ids, judgment_dict, n=10)



            """ before optimization (each computing use a loop) """
            # calculate the precision of this query
            # sum_precision += calculate_precision(sorted_doc_ids, judgment_dict)
            # calculate the recall of this query
            # sum_recall += calculate_recall(sorted_doc_ids, judgment_dict, rel_count)
            # calculate the P@10 of this query
            # sum_p_at10 += calculate_P_at_n(sorted_doc_ids, n=10, judgment_dict=judgment_dict)
            # calculate the R-precision of this query
            # sum_R_precision += calculate_R_precision(sorted_doc_ids, judgment_dict, rel_count)
            # calculate the AP of this query
            # sum_MAP += calculate_AP(sorted_doc_ids, judgment_dict, rel_count)
            # calculate the bpref of this query
            # sum_bpref += calculate_bpref(sorted_doc_ids, judgment_dict, rel_count)
            # calculate the NDCG@10 of this query (NDCG@10 is very commonly used)
            # sum_NDCG += calculate_NDCG_at_n(sorted_doc_ids, judgment_dict, n=10)
            
            # write the result of this query into the output file
            for index, doc_id in enumerate(sorted_doc_ids):
                # each line: queryID, "Q0", doc_id, rank, similarity_score, UCD stu_number
                output_file.write("{} Q0 {} {} {:.4f} {}\n".format(query_id, doc_id, index + 1, sim_dict[doc_id], UCD_STUDENT_NUMBER))


    # calculate the average scores in each metric
    query_len = len(standard_queries)   # how many queries have been searched for evaluation
    avg_precision = sum_precision / query_len
    avg_recall = sum_recall / query_len
    avg_p_at10 = sum_p_at10 / query_len
    avg_R_precision = sum_R_precision / query_len
    avg_MAP = sum_MAP / query_len
    avg_bpref = sum_bpref / query_len
    avg_NDCG = sum_NDCG / query_len

    # print out the average result
    print("Evaluation results: ")
    print("{:<14}{:.4f}".format("Precision:", avg_precision))
    print("{:<14}{:.4f}".format("Recall:", avg_recall))
    print("{:<14}{:.4f}".format("P@10:", avg_p_at10))
    print("{:<14}{:.4f}".format("R-precision:", avg_R_precision))
    print("{:<14}{:.4f}".format("MAP:", avg_MAP))
    print("{:<14}{:.4f}".format("bpref:", avg_bpref))
    print("{:<14}{:.4f}".format("NDCG:", avg_NDCG))


''' ----------------------------------------------- Starter Configurations ----------------------------------------------- '''

def init_bm25():
    """
        This function initializes the bm25 model.
    """
    print(' - Loading BM25 index from file, please wait...')

    start_time = time.process_time()

    # read in the stopwords
    load_stopwords()
    # load in index info
    load_index()

    end_time = time.process_time()

    print(' - Index loading finished. Time consumption is {} second.'.format(end_time - start_time))


def start_manual():
    """
        This is the starter for manual mode.
        When user selected manual mode in their command line, 
        this starter would be executed.
        Then the user can enter queries manually.
    """
    # initialize the bm25 model
    init_bm25()
    
    # start to listen to the user input
    listen_query()


def start_evaluation():
    """
        This is the starter for evaluation mode.
        When user selected evaluation mode in their command line, 
        this starter would be executed.
        Then some queries will be executed automatically to evaluate this bm25 model.
        When finished, the user will get a list of scores of different evaluation metrics.
    """
    # initialize the bm25 model
    init_bm25()
    # load standard queries for evaluation
    load_queries()
    # load relevance judgments
    load_relevance_judgments()

    # start the process of evaluation
    print(' - Evaluation begin, please wait...')
    start_time = time.process_time()
    begin_evaluation()
    end_time = time.process_time()
    print(' - Evaluation finished. Time consumption is {} second.'.format(end_time - start_time))


def start():
    """
        This is the starter of this program, which is 
        used to check the user input in the command line.
        Then different modes will be started according to the user selection.
    """
    # get parameters from command line
    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv, "m:")

    except getopt.GetoptError as e:
        print(e)
        sys.exit(1)

    # check if there are options given in the command line
    if len(opts) > 0:

        # check each option to get the running mode
        mode = ""
        for opt, arg in opts:
            if opt == '-m':
                mode = arg

        # check the running mode
        if mode == "manual":                                                
            print(" - Initialized with 'manual' mode.")
            start_manual()

        elif mode == "evaluation":
            print(" - Initialized with 'evaluation' mode")
            start_evaluation()

        else:
            print("No such mode! 'manual' and 'evaluation' only.")
            sys.exit(0)

    else:
        # if no options given, we start with the "manual" as default one
        print(" - Initialized with 'manual' mode as default.")
        start_manual()


''' ----------------------------------------------- Starting Entrance ----------------------------------------------- '''

if __name__ == '__main__':
    start()
