# Relevant Papers

## Statistical Machine Translation: IBM Models 1 and 2
	* language model for probability of an english sentence
	* translation model for conditional probability of a french/english pair 
	* Alignments
		* Set length of french sentence to be fixed
		* Alignment variable for each French word to some word in English 
		 	sentence
		* Sum over all alignment variables to marginalize them out
	* IBM Model 2
		* parameters:
			* t(f|e) = conditional probability of generating French word f 
				from english word e
			* q(j|i,l,m) = probability of alignment variable a_i taking the
				value j, conditioned on lengths l and m of english and 
				french sentences
			* each french word has two terms associated with it: choice
			 	of alignment variable specifying which english word aligned
			 	to and then the choice of the french word itself for the given
			 	english word
			* strong independence assumption--distribution of alignment
				variable only depends on lengths of english and french 
				sentences
			* models good because lexical probabilities t(f|e) directly
				used in translation systems and are of direct use in building
				modern translation systems
	* Parameter Estimation with Fully Observed Data
		* don't know underlying alignment for each training example

* http://www.socher.org/index.php/Main/ImprovingWordRepresentationsViaGlobalContextAndMultipleWordPrototypes