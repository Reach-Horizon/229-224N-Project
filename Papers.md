# Relevant Papers


* http://www.socher.org/index.php/Main/ImprovingWordRepresentationsViaGlobalContextAndMultipleWordPrototypes

-------------------------

## Machine Learning in Automated Text Categorization

* semantic representations of words (through word vectors) to determine relatedness of terms

## Finding Question-Answer Pairs from Online Forums

* employ method of detecting question-answer pairs for a given question posted and subsequent responses given
* Question detection to automatically be able to detect questions in forums
* Given question, detect answer passages within same forum thread
* Answer detection may be thought of as standard documet retrieval problem, treating each response as a possible document to provide for a given query; however this method does not account for distance of candidate answer from question

# Question Detection
* Use labeled sequential patterns
* pattern of form LHS -> c where LHS is a sequence and c is a class label
* POS tag sentences keeping keywords but preserving other words, transforming each sentence into a database tuple

# Answer Detection
