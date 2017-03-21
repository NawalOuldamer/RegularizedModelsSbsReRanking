package org.mrim.utils;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;


public class Store {
	private Double lambda;
	private String MODEL;
	private String MODEL_PATH;

	public Store(double lambda,String MODEL,String MODEL_PATH ){
		this.lambda = lambda;
		this.MODEL = MODEL;
		this.MODEL_PATH = MODEL_PATH;
	}
	
	public Store(){
		
	}
	
	public void storeDocumentModel(List<Map.Entry<String, Double>> list, String docId, String queryID) throws IOException{	
		if(!new File(this.MODEL_PATH+"/"+queryID+"/"+this.lambda).exists()){			
			new File(this.MODEL_PATH+"/"+queryID+"/"+this.lambda).mkdirs();
		}
		FileWriter file = new FileWriter(this.MODEL_PATH+"/"+queryID+"/"+this.lambda+"/"+this.MODEL,true);
		file.write(docId+" ");
		for (Iterator<Entry<String, Double>> iterator = list.iterator(); iterator.hasNext();) {
			Entry<String, Double> entry = iterator.next();
			file.write(entry.getKey()+":"+entry.getValue()+" ");				
		}
		file.write("\n");
		file.close();
	}
	
}