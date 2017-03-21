package org.mrim.w2v;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.nd4j.linalg.api.ndarray.INDArray;


/**
 * @author ould
 *
 */
public class TermVector {

	
	public WordVectors WORDS_VECTORS;

	public TermVector(WordVectors WORDS_VECTORS){
		this.WORDS_VECTORS = WORDS_VECTORS;
	}

	public INDArray buildTermVector(String term){
		INDArray termVector = null;
		List<String> cleanList = new ArrayList<String>(); // Check if term are in model w2v
		for (Iterator<String> iterator = Arrays.asList(term.split(" ")).iterator(); iterator.hasNext();) {
			String t = iterator.next();
			if(WORDS_VECTORS.hasWord(t)){
				cleanList.add(t);
			}	
		}
		try {
			if(!cleanList.isEmpty()){
				termVector = WORDS_VECTORS.getWordVectorsMean(cleanList);
			}	
		} catch (Exception e) {
			System.err.println("Query Vector for query: "+term+ " is NULL");
		}
		return termVector;
	}

}