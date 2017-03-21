/**
 * 
 */
package org.mrim.models.document;

import java.io.IOException;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.mrim.utils.Store;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.terrier.structures.Index;

/**
 * @author ould
 *
 */
public class DocumentQueryRegularizedMixtureModel {

	/**
	 * @param args
	 */
	/* Parameters*/
	private int numberOfIterations;
	private double probThreshold;
	private double lambda;
	private String MODEL;

	/* Paths */
	private String PATH_STORE_MODEL;
	private String TERRIER_DOC_MAP_PATH;
	private String WORD2VEC_MODEL_PATH;
	private String PATH_RES_TERRIER_FILE;
	private String INDEX_PATH;


	/* Generated */
	private HashMap<String, HashMap<String,Double>> DOCUMENTS_MODELS = new HashMap<>();
	private HashMap<String, HashMap<String,Double>> DOCUMENTS_TF_MODELS = new HashMap<>();
	private HashMap<String, HashSet<String>> QUERY_DOCUMENTS = new HashMap<>();
	private HashMap<String, INDArray> QUERIES_VECTORS = new HashMap<>();
	private WordVectors WordVectors;
	private Index INDEX;
	public HashMap<String, String> MAP_TERRIER_DOC = new HashMap<String, String>();


	public DocumentQueryRegularizedMixtureModel(int numberOfIterations,double probThreshold,double lambda,String PATH_STORE_MODEL,String MODEL,
			String PATH_RES_TERRIER_FILE, String WORD2VEC_MODEL_PATH,String INDEX_PATH,String TERRIER_DOC_MAP_PATH
			){
		this.numberOfIterations = numberOfIterations;
		this.probThreshold = probThreshold;
		this.lambda = lambda;
		this.PATH_STORE_MODEL = PATH_STORE_MODEL;
		this.MODEL = MODEL;
		this.PATH_RES_TERRIER_FILE = PATH_RES_TERRIER_FILE;
		this.WORD2VEC_MODEL_PATH =WORD2VEC_MODEL_PATH;
		this.INDEX_PATH =  INDEX_PATH;
		this.TERRIER_DOC_MAP_PATH = TERRIER_DOC_MAP_PATH;
		
	}

	public void runDQRMM() throws IOException{
		initialization();
		Store store = new Store(this.lambda, MODEL, PATH_STORE_MODEL);		
		for (Map.Entry<String, HashSet<String>> entry : this.QUERY_DOCUMENTS.entrySet()) {
			String queryID = entry.getKey();
			for (Iterator<String> iterator = entry.getValue().iterator(); iterator.hasNext();) {
				String docID = iterator.next();
				Estimation estimate = new Estimation(numberOfIterations, probThreshold, lambda, INDEX, WordVectors);
				estimate.initialization(this.DOCUMENTS_MODELS.get(docID), DOCUMENTS_TF_MODELS.get(docID), this.QUERIES_VECTORS.get(queryID));
				estimate.estimate();
				HashMap<String, Double> docModel = new HashMap<>();
				docModel = estimate.getDocumentModel();

				/** Save Document Model**/
				List<Entry<String, Double>> sortedDocumentModel = sortByValues(docModel,true);  // sort by high value of term weight			
				List<Map.Entry<String, Double>> DocumentModel = getTopK(sortedDocumentModel.size(), sortedDocumentModel);
				store.storeDocumentModel(DocumentModel, docID, queryID);
			}
		}			
	}

	public void initialization() throws IOException{
		InitModel init = new InitModel(INDEX_PATH, TERRIER_DOC_MAP_PATH, WORD2VEC_MODEL_PATH, PATH_RES_TERRIER_FILE);
		init.loadDocumentsQuery();
		this.QUERY_DOCUMENTS = init.getQUERY_DOCUMENTS();
		init.loadIndex();
		this.INDEX = init.getIndex();
		init.loadTerrierIDMapping();
		this.MAP_TERRIER_DOC= init.getMAP_TERRIER_DOC();
		init.loadW2VModel();
		this.WordVectors = init.getWordVectors();
	}

	public List<Map.Entry<String, Double>> getTopK(Integer k, List<Map.Entry<String, Double>> map) {
		HashMap<String, Double> m = new HashMap<String, Double>();
		for (Iterator<Entry<String, Double>> iterator = map.iterator(); iterator.hasNext();) {
			Entry<String, Double> entry = iterator.next();
			m.put(entry.getKey(), entry.getValue());
		}
		List<Map.Entry<String, Double>> sorted = sortByValues(m, false);
		k = k < sorted.size() ? k : sorted.size();
		return sorted.subList(0, k);
	} 
	
	public List<Map.Entry<String, Double>> sortByValues(Map<String, Double> unsortMap, final boolean order) {
		List<Map.Entry<String, Double>> list = new LinkedList<Map.Entry<String, Double>>(unsortMap.entrySet());
		Collections.sort(list, new Comparator<Map.Entry<String, Double>>() {
			public int compare(Map.Entry<String, Double> o1,
					Map.Entry<String, Double> o2) {
				if (order) {
					return o1.getValue().compareTo(o2.getValue());
				} else {
					return o2.getValue().compareTo(o1.getValue());
				}
			}
		});
		return list;
	}
}
