import tempfile
import os
import subprocess
import pandas as pd
import re
import threading
from threading import Lock

def preprocess_text(text):
    return re.sub(r' +', ' ',text.lower().replace("'", '').replace('"', '').replace('\n', ' ').replace('\t', ' ').replace('\r', ' '))

def run_udpipe2_client(text: str):
    text = preprocess_text(text)
    """Run the UDPipe2 client on a text and return the result as a string."""
    # Create a temporary file with the text.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False) as inputFile:
        inputFile.write(text)
        inputFile.close()
        # Run the UDPipe2 client on the temporary file.
        # The output is written to a file.
        outputFileName = inputFile.name + ".out"
        subprocess.Popen(f'python3 udpipe2_client.py --model portuguese-gsd-ud-2.10-220711 --input generic_tokenizer --tokenizer ranges --tagger 1 --parser 1 --outfile {outputFileName} {inputFile.name}', shell=True).wait()
        
        # Read the output file.
        with open(outputFileName, "r", encoding="utf-8") as outputFile:
            result = outputFile.read()
        
        # Delete the temporary files.
        os.remove(inputFile.name)
        os.remove(outputFileName)
        
    return result

def get_lemma(word, lemma):
    return lemma if lemma != '_' else word

def joinSubjectOrObject(wordAdjList: list, subjectOrObjectIndex: int, subjectOrObjectWord: str, subjectOrObjectLemma: str, subjectOrObjectXpos):
    #   nsubj/obj
    #    /      \
    # rel1     rel2
    if subjectOrObjectXpos == 'PRON':
        return; # ignore pronouns

    bannedSubjectOrObjects = {'-', '––', '—', '‘', '’', '“', '”', '♪', '‍♂', '–', '+', '++', '+++', '+++‘'}

    lemma = get_lemma(subjectOrObjectWord, subjectOrObjectLemma)
    if lemma in bannedSubjectOrObjects:
        return

    relationsToAppend = {'amod', 'nmod', 'nummod', 'case', 'appos', 'flat'}
    strComponents = [(subjectOrObjectIndex, lemma)]

    def helper(index: int):
        nonlocal wordAdjList
        nonlocal strComponents
        nonlocal relationsToAppend
        
        for auxIndex, auxWord, auxLemma, _, auxRelation in wordAdjList[index]:
            lemma = get_lemma(auxWord, auxLemma)
            if auxRelation in relationsToAppend and lemma not in bannedSubjectOrObjects:
                strComponents.append((auxIndex, lemma))
                helper(auxIndex)
    
    # O(w), worst case is the tree is a list with all words being of a relation to append
    helper(subjectOrObjectIndex)

    # O(w * log(w) + len(subj)) to sort the list + join the words
    subj = ' '.join([word for _, word in sorted(strComponents, key=lambda x: x[0])])
    return subj

def joinVerb(wordAdjList: list, verbIndex: int, verbWord: str, verbLemma: str):
    #    verb
    #    /   \
    #  rel1  verb
    #          \
    #         rel2
    nonVerbRelationsToAppend = {'advmod'}
    verbRelationsToAppend = {'xcomp', 'conj'}

    strComponents = [(verbIndex, get_lemma(verbWord, verbLemma))]
    def findAllVerb(index: int):
        nonlocal wordAdjList
        nonlocal verbRelationsToAppend
        nonlocal strComponents

        for auxIndex, auxWord, auxLemma, auxXpos, relation in wordAdjList[index]:
            if auxXpos == 'VERB' and relation in verbRelationsToAppend:
                strComponents.append((auxIndex, get_lemma(auxWord, auxLemma)))
                findAllVerb(auxIndex)    
        
    findAllVerb(verbIndex)

    # find all non-verb relations to append
    verbIndexes = [index for index, _ in strComponents]
    for i in verbIndexes:
        for auxIndex, auxWord, auxLemma, auxXpos, auxRelation in wordAdjList[i]:
            if auxXpos != 'VERB' and auxRelation in nonVerbRelationsToAppend:
                strComponents.append((auxIndex, get_lemma(auxWord, auxLemma)))

    # O(w * log(w)) to sort the list
    verb = [word for _, word in sorted(strComponents, key=lambda x: x[0])]   
    verb = ' '.join(verb) 

    return verb, min(verbIndexes), max(verbIndexes)

def parse_udpipe2_output(text: str):
    """Parse the UDPipe2 output and return a list of tuples."""
    # get each sentence in a list (they are separated by a blank line)
    sentences = text.rstrip().split('\n\n')
    
    tuples = []
    for sentence in sentences: # O(n * w^3)
        # Remove every line that does not start with a number.
        sentence = [line for line in sentence.splitlines() if line and line[0].isdigit()]

        # get the word, lemma, xpos, head, and relation (remove lines with ranges as index)
        words = [line.split('\t') for line in sentence]
        words_tuples = [(int(w[0]), w[1], w[2], w[4], int(w[6]) if w[6] != '_' else 0, w[7]) for w in words if w[0].isdigit()]

        # create a adjlist from the words_tuples
        # adjList[i] = every word that has i as head
        wordAdjList = [[] for _ in range(len(words_tuples) + 1)]
        for index, word, lemma, xpos, head, relation in words_tuples:
            wordAdjList[head].append((index, word, lemma, xpos, relation))

        # O(w^3 * log(w))
        # Form tuples that follow this pattern
        #     verb
        #    /    \
        #  nsubj   obj
        nsubj = ''
        verb = ''
        obj = ''
        for verbIndex, verbWord, verbLemma, verbXpos, _, _ in words_tuples:
            if verbXpos != 'VERB': continue
            verb, firstVerbIndex, lastVerbIndex = joinVerb(wordAdjList, int(verbIndex), verbWord, verbLemma)
            if not verb: continue
            for subjIndex, subjWord, subjLemma, subjXpos, subjRelation in wordAdjList[int(firstVerbIndex)]:
                if subjRelation != 'nsubj': continue
                nsubj = joinSubjectOrObject(wordAdjList, subjIndex, subjWord, subjLemma, subjXpos)
                if not nsubj: continue
                for objIndex, objWord, objLemma, objXpos, objRelation in wordAdjList[int(lastVerbIndex)]:
                    if objRelation != 'obj': continue
                    obj = joinSubjectOrObject(wordAdjList, objIndex, objWord, objLemma, objXpos)
                    if not obj: continue
                    tuples.append((nsubj, verb, obj))

            
            # reset variables
            verb = ''
            nsubj = ''
            obj = ''

    return tuples

def run_with_threads(number_of_threads, texts, name):
    lock = Lock()
    def process_texts_thread(lock, start, end):
        nonlocal results
        nonlocal texts
        partial_results = []
        for text in texts[start:end]:
            udpipe2_output = run_udpipe2_client(text)
            
            # Parse the UDPipe2 output.
            partial_results.extend(parse_udpipe2_output(udpipe2_output))

        lock.acquire()
        results.extend(partial_results)
        lock.release()

    number_of_texts = len(texts)
    texts_per_thread = number_of_texts // number_of_threads

    threads = []
    results = []

    for i in range(number_of_threads):
        start = i * texts_per_thread
        end = (i + 1) * texts_per_thread if i < number_of_threads - 1 else number_of_texts
        thread = threading.Thread(target=process_texts_thread, args=(lock, start, end))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    result_df = pd.DataFrame(results, columns=['subject', 'verb', 'object'])
    result_df.drop_duplicates(inplace=True)
    result_df.dropna(inplace=True)
    result_df.sort_values(by=['subject', 'verb', 'object'], inplace=True)
    result_df.reset_index(drop=True, inplace=True)
    result_df.to_csv(f'../data/csvs/{name}.csv', index=False)

def break_text_into_sentences(text: str):
    def validate_sentence(sentence: str):
        # remove sentences that end with ? or are too short
        return len(sentence) > 30 and sentence[-1] != '?'

    # split sentences by punctuation (., !, ?) and write each sentence in a new line
    sentences = []
    for sentence in re.split(r'([.!?])', text):
        if validate_sentence(sentence):
            sentences.append(sentence.strip())
    
    return sentences

def run_sequential(text): 
    udpipe2_output = run_udpipe2_client(text)
    tuples = parse_udpipe2_output(udpipe2_output)
    result_df = pd.DataFrame(tuples, columns=['subject', 'verb', 'object'])
    result_df.drop_duplicates(inplace=True)
    result_df.dropna(inplace=True)
    result_df.sort_values(by=['subject', 'verb', 'object'], inplace=True)
    result_df.reset_index(drop=True, inplace=True)

    # convert back into array of tuples (subject, verb, object)
    tuples = [tuple(x) for x in result_df.to_numpy()]
    return tuples
