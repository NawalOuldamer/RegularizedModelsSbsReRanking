/**
 * 
 */
package org.mrim.models.document;

import java.util.HashMap;
import java.util.Map;

import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.mrim.w2v.TermVector;
import org.mrim.w2v.VectorSimilarity;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.terrier.structures.Index;
import org.terrier.structures.Lexicon;
import org.terrier.structures.LexiconEntry;

/**
 * @author ould
 *
 */
public class Estimation {

	private int numberOfIterations;
	private double probThreshold;
	private double lambda;
	private Index INDEX;
	private HashMap<String, Double> DocumentModel;
	private HashMap<String, Double> DocumentTF;
	private INDArray queryVector;
	private WordVectors WORDS_VECTORS;
	TermVector termVec;
	VectorSimilarity vecSim;
	/**
	 * @param args
	 */
	
	
	
	public Estimation(int numberOfIterations,double probThreshold,double lambda,Index  INDEX, WordVectors WORDS_VECTORS){
		this.numberOfIterations = numberOfIterations;
		this.probThreshold = probThreshold;
		this.lambda = lambda;
		this.INDEX = INDEX;
		this.WORDS_VECTORS = WORDS_VECTORS;
	}

	public void initialization(HashMap<String, Double> DocumentModel,HashMap<String, Double> DocumentTF, INDArray queryVector){
		/** initialization step **/
		this.DocumentModel = new HashMap<>();
		this.DocumentTF = new HashMap<>();
		this.DocumentModel = DocumentModel;
		this.DocumentTF = DocumentTF;
		this.queryVector = null;
		this.queryVector = queryVector;
		termVec = new TermVector(this.WORDS_VECTORS);
		vecSim = new VectorSimilarity(this.WORDS_VECTORS);
	}
	
	public void estimate(){
		for (int i = 0; i < this.numberOfIterations ; i++) {
			this.E_Step();    
			this.M_Step();	
		}
	}
	
	public void E_Step(){
		for (Map.Entry<String, Double> entry : this.DocumentModel.entrySet()) {
			String term = entry.getKey();
			Double classicPr = lambda  * (this.DocumentModel.get(term)) / (lambda * (this.DocumentModel.get(term)) + ((1-lambda) * getTermCollectionPR(term)));			
			INDArray termVect = termVec.buildTermVector(term);
			Double queryPr = vecSim.estimateVectorSimilarity(this.queryVector, termVect);
			Double tf =  this.DocumentTF.get(term) ;
			Double newPr = queryPr * tf * classicPr; 
			// update new probability
			this.DocumentModel.put(term, newPr);
		}
	}
	
	public void M_Step(){		
		double normalization_Part =  this.DocumentModel.values().stream().mapToDouble(Number::doubleValue).sum();
		HashMap<String, Double> newDocumentModel = new HashMap<String, Double>();
		for (Map.Entry<String, Double> entry : this.DocumentModel.entrySet()) {
			String term = entry.getKey();
			double newPr = this.DocumentModel.get(term)/normalization_Part;
			if(newPr>=this.probThreshold){
				newDocumentModel.put(term, newPr);
			}
		}
		
		this.DocumentModel.clear();
		this.DocumentModel = newDocumentModel;
	}
	public double getTermCollectionPR(String term){
		Lexicon<String> lex = INDEX.getLexicon();
		LexiconEntry le = lex.getLexiconEntry(term);
		double p = le == null
				?  0.0d
						: (double) le.getFrequency() / INDEX.getCollectionStatistics().getNumberOfTokens();
		return p;
	}
	
	public int getNumberOfIterations() {
		return numberOfIterations;
	}

	public void setNumberOfIterations(int numberOfIterations) {
		this.numberOfIterations = numberOfIterations;
	}

	public double getProbThreshold() {
		return probThreshold;
	}

	public void setProbThreshold(double probThreshold) {
		this.probThreshold = probThreshold;
	}

	public double getLambda() {
		return lambda;
	}

	public void setLambda(double lambda) {
		this.lambda = lambda;
	}

	public Index getINDEX() {
		return INDEX;
	}

	public void setINDEX(Index iNDEX) {
		INDEX = iNDEX;
	}

	public HashMap<String, Double> getDocumentModel() {
		return DocumentModel;
	}

	public void setDocumentModel(HashMap<String, Double> documentModel) {
		DocumentModel = documentModel;
	}

	public HashMap<String, Double> getDocumentTF() {
		return DocumentTF;
	}

	public void setDocumentTF(HashMap<String, Double> documentTF) {
		DocumentTF = documentTF;
	}

	
}
