package org.mrim.models.document;

import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;

import org.apache.log4j.Logger;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.terrier.structures.Index;

public class RunDocumentModel {
	
	
	
	/**
	 * @param args
	 */
	public static String INDEX_PATH="";
	public static String WORD2VEC_MODEL_PATH;
	public static String TERRIER_DOC_MAP_PATH;
	public static String PATH_RES_TERRIER_FILE;
	
	
	public static Logger logger = Logger.getRootLogger();

	public static Index  index;
	public static WordVectors wordVectors;
	public static HashMap<String, HashSet<String>> QUERY_DOCUMENTS = new HashMap<>();
	public static HashMap<String, String> MAP_TERRIER_DOC = new HashMap<String, String>();
	private static String MODEL;
	private static double lambda;
	private static double probThreshold;
	private static int numberOfIterations;
	private static String PATH_STORE_MODEL;

	public static void main(String[] args) throws IOException {
		if (args.length == 0 || args[0].equals("--help") ) {
			System.out.print("Usage: " ); usage();
		}
		else {
			for (int i = 0; i < args.length; i = i+2) {
				if(args[i].equals("--build-model")){
					MODEL = args[i+1];
				}
				if(args[i].equals("--lambda")){
					lambda = Double.parseDouble(args[i+1]);
				}
				if(args[i].equals("--probThreshold")){
					probThreshold = Double.parseDouble(args[i+1]);
				}
				if(args[i].equals("--numberOfIterations")){
					numberOfIterations = Integer.parseInt(args[i+1]);
				}
				if(args[i].equals("--input-index-terrier-path")){
					INDEX_PATH = args[i+1];
				}
				if(args[i].equals("--input-terrier-doc-map")){
					TERRIER_DOC_MAP_PATH = args[i+1];
				}
				if(args[i].equals("--input-path-store-model")){
					PATH_STORE_MODEL = args[i+1];
				}				
				if(args[i].equals("--output-word2vec-model-path")){
					WORD2VEC_MODEL_PATH = args[i+1];
				}
				if(args[i].equals("--input-path-terrier-intial-results")){
					PATH_RES_TERRIER_FILE = args[i+1];
				}
			}
			
			if(MODEL.equalsIgnoreCase("QRDLM")){
				logger.info("Build");
				logger.info("MODEL: " + MODEL);
				logger.info("lambda: " + lambda);
				logger.info("probThreshold: "+probThreshold);
				logger.info("numberOfIterations: "+numberOfIterations);
				logger.info("INDEX_PATH: "+INDEX_PATH);
				logger.info("TERRIER_DOC_MAP_PATH: "+TERRIER_DOC_MAP_PATH);
				logger.info("WORD2VEC_MODEL_PATH: "+WORD2VEC_MODEL_PATH);
				logger.info("PATH_RES_TERRIER_FILE: "+PATH_RES_TERRIER_FILE);
				/**Initialization of the model**/
				initDocumentModel();
				/**Estimate the model **/
				DocumentQueryRegularizedMixtureModel docQueryRegMixModel = new DocumentQueryRegularizedMixtureModel(numberOfIterations, probThreshold, lambda, 
						PATH_STORE_MODEL, MODEL, PATH_RES_TERRIER_FILE, WORD2VEC_MODEL_PATH, INDEX_PATH, TERRIER_DOC_MAP_PATH);
				docQueryRegMixModel.runDQRMM();
				
			}
		}
		
	}
	public static void usage(){
		System.out.println("	--build-model [QRDLM or TRDLM] QRDLM= QueryRegularizedModel, TRDLM: TagRegularizedModel ");
		System.out.println("	--lambda [value]  ");
		System.out.println("	--probThreshold [value] ");
		System.out.println("	--numberOfIterations [value] ");
		System.out.println("	--output-word2vec-model-path [FILENAME] This option specifies a file that contains word2vec model");
		System.out.println("	--input-index-terrier-path [FILENAME] terrier index path ");
		System.out.println("	--input-terrier-doc-map [FILENAME] terrier map document path ");
		System.out.println("	--input-path-store-model [FILENAME] file where the outpu models where be store");


	}

	public static void initDocumentModel() throws IOException{
		logger.info("Model Initialization");
		InitModel init = new InitModel(INDEX_PATH,TERRIER_DOC_MAP_PATH,WORD2VEC_MODEL_PATH,PATH_RES_TERRIER_FILE);
		init.loadIndex();
		index = init.getIndex();
		init.loadDocumentsQuery();
		QUERY_DOCUMENTS = init.getQUERY_DOCUMENTS();
		init.loadTerrierIDMapping();
		MAP_TERRIER_DOC = init.getMAP_TERRIER_DOC();		
		init.loadW2VModel();
		wordVectors = init.getWordVectors();	
		logger.info("Model Initialization : Done !");
	}


}
