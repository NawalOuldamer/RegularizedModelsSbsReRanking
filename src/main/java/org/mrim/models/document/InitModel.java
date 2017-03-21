/**
 * 
 */
package org.mrim.models.document;
/**
 * 
 */
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Scanner;
import org.apache.log4j.Logger;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.mrim.w2v.LoadTextVector;
import org.terrier.structures.Index;


/**
 * @author Nawal OULD-AMER
 *
 */
public class InitModel {

	/**
	 * @param args
	 * 
	 */

	public String INDEX_PATH="";
	public String WORD2VEC_MODEL_PATH;
	public String TERRIER_DOC_MAP_PATH;
	public String PATH_RES_TERRIER_FILE;

	static int numberOfDoc = 0;
	public Logger logger = Logger.getRootLogger();
	
	
	
	public Index  index;
	public WordVectors wordVectors;
	HashMap<String, HashSet<String>> QUERY_DOCUMENTS = new HashMap<>();
	public HashMap<String, String> MAP_TERRIER_DOC = new HashMap<String, String>();
	
	
	public InitModel(){
	}
	
	public InitModel(String INDEX_PATH,String TERRIER_DOC_MAP_PATH,String WORD2VEC_MODEL_PATH,String PATH_RES_TERRIER_FILE){
		this.INDEX_PATH = INDEX_PATH;
		this.TERRIER_DOC_MAP_PATH = TERRIER_DOC_MAP_PATH;
		this.WORD2VEC_MODEL_PATH = WORD2VEC_MODEL_PATH;
		this.PATH_RES_TERRIER_FILE = PATH_RES_TERRIER_FILE;
	}


	/**
	 * Load index collection using TERRIER
	 */
	protected void loadIndex(){
		long startLoading = System.currentTimeMillis();
		index = Index.createIndex(INDEX_PATH, "data");
		if(index == null)
		{
			logger.fatal("Failed to load index. Perhaps index files are missing");
		}
		long endLoading = System.currentTimeMillis();
		if (logger.isInfoEnabled())
			logger.info("time to intialise index : " + ((endLoading-startLoading)/1000.0D));
	}	
	
	public void loadW2VModel() throws IOException{
		InputStream stream = new FileInputStream(WORD2VEC_MODEL_PATH);
		LoadTextVector load_model = new LoadTextVector();
		wordVectors = load_model.loadTxtVectors(stream, false);
	}

	/**
	 * Load terrier map	 *
	 */
	public void loadTerrierIDMapping(){
		try {
			Scanner s = new Scanner(new File(TERRIER_DOC_MAP_PATH));
			while (s.hasNextLine()) {
				String [] line = s.nextLine().split(" ");
				MAP_TERRIER_DOC.put(line[0], line[1]);				
			}
			s.close();

		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}
	
	public void loadDocumentsQuery() throws FileNotFoundException{
		Scanner sc = new Scanner(new File(PATH_RES_TERRIER_FILE));
		while (sc.hasNextLine()) {
			String line = (String) sc.nextLine();
			String [] v = line.split(" ");
			if(QUERY_DOCUMENTS.containsKey(v[0])){
				HashSet<String> l = new HashSet<>();
				l.addAll(QUERY_DOCUMENTS.get(v[0]));
				l.add((v[2]));
				QUERY_DOCUMENTS.put(v[0], l);
			}
			else {
				HashSet<String> l = new HashSet<>();
				l.add((v[2]));
				QUERY_DOCUMENTS.put(v[0], l);
			}			
		}
		sc.close();
	}
	
	public Index getIndex() {
		return index;
	}

	public void setIndex(Index index) {
		this.index = index;
	}

	public WordVectors getWordVectors() {
		return wordVectors;
	}

	public void setWordVectors(WordVectors wordVectors) {
		this.wordVectors = wordVectors;
	}

	public HashMap<String, HashSet<String>> getQUERY_DOCUMENTS() {
		return QUERY_DOCUMENTS;
	}

	public void setQUERY_DOCUMENTS(HashMap<String, HashSet<String>> qUERY_DOCUMENTS) {
		QUERY_DOCUMENTS = qUERY_DOCUMENTS;
	}

	public HashMap<String, String> getMAP_TERRIER_DOC() {
		return MAP_TERRIER_DOC;
	}

	public void setMAP_TERRIER_DOC(HashMap<String, String> mAP_TERRIER_DOC) {
		MAP_TERRIER_DOC = mAP_TERRIER_DOC;
	}

	
}