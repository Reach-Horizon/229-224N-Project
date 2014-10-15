# Relevant Papers


* http://www.socher.org/index.php/Main/ImprovingWordRepresentationsViaGlobalContextAndMultipleWordPrototypes

-------------------------

## Machine Learning in Automated Text Categorization

* semantic representations of words (through word vectors) to determine relatedness of terms

## Finding Question-Answer Pairs from Online Forums (Cong et al.)

* employ method of detecting question-answer pairs for a given question posted and subsequent responses given
* Question detection to automatically be able to detect questions in forums
* Given question, detect answer passages within same forum thread
* Answer detection may be thought of as standard documet retrieval problem, treating each response as a possible document to provide for a given query; however this method does not account for distance of candidate answer from question
* Question Detection
  * Use labeled sequential patterns
  * pattern of form LHS -> c where LHS is a sequence and c is a class label
  * POS tag sentences keeping keywords but preserving other words, transforming each sentence into a database tuple
* Answer Detection
  * develop a graph of related answers using * KL-divergence language model *

## An Intelligent Discussion-Bot for Answering Student Queries in Threaded Discussions (Feng et al.)

* TextTiling NLP tool to segment every document into semanticaly-related tiles and hence process tile units; 
* http://people.ischool.berkeley.edu/~hearst/research/tiling.html
* Student question -> Feature Extraction -> Student Interest Matching -> Supplemental Course Documents and Archived Threaded Discussions -> Answer Generation -> Reply Posted to DB
* human judge manually deems quality of an answer provided, based on a number of categories

## Extracting Chatbot Knowledge from Online Discussion Forums (Huang et al.)

* detect direct reply to first post in thread
* use a cascaded model:
  * apply SVM classifier to candidate RR to identify RR (direct reply) of a thread
  * filter out noneligible RR
  * identifying RR seen as a binary classification problem
  * features used listed in paper
 
## Extracting Chatbot Knowledge from Online Discussion Forums

* divide responses into text tiles
* cluster into topics
* sentential segments taken from each cluster to form summary
* LIBSVM, SVMLiTE

## Bridging the Lexical Chasm: Statistical Approaches to Answer Finding

* MT model to question-answer finding
* source = answers, target = questions
* translation model learns how answer-words translate to question words, bridging lexical chasm
