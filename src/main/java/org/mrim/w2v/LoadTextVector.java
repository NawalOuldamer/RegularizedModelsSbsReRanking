package org.mrim.w2v;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.reader.impl.BasicModelUtils;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectorsImpl;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import lombok.NonNull;

/**
 * @author ould
 *
 */
public class LoadTextVector {
	
	public static WordVectors wordVectors;



	private static final String whitespaceReplacement = "_Az92_";


	@SuppressWarnings({ "rawtypes", "deprecation" })
	public WordVectors loadTxtVectors(@NonNull InputStream stream, boolean skipFirstLine) throws IOException {
		AbstractCache<VocabWord> cache = new AbstractCache.Builder<VocabWord>().build();

		BufferedReader reader = new BufferedReader(new InputStreamReader(stream));
		String line = "";
		List<INDArray> arrays = new ArrayList<INDArray>();

		if (skipFirstLine)
			reader.readLine();

		while((line = reader.readLine()) != null) {
			String[] split = line.split(" ");
			String word = split[0].replaceAll(whitespaceReplacement, " ");
			VocabWord word1 = new VocabWord(1.0, word);

			word1.setIndex(cache.numWords());

			cache.addToken(word1);

			cache.addWordToIndex(word1.getIndex(), word);

			cache.putVocabWord(word);

			/*
            INDArray row = Nd4j.create(Nd4j.createBuffer(split.length - 1));
            for (int i = 1; i < split.length; i++) {
                row.putScalar(i - 1, Float.parseFloat(split[i]));
            }
			 */

			float[] vector = new float[split.length - 1];

			for (int i = 1; i < split.length; i++) {
				vector[i-1] = Float.parseFloat(split[i]);
			}

			INDArray row = Nd4j.create(vector);

			arrays.add(row);
		}

		InMemoryLookupTable<VocabWord> lookupTable = (InMemoryLookupTable<VocabWord>) new InMemoryLookupTable.Builder<VocabWord>()
				.vectorLength(arrays.get(0).columns())
				.cache(cache)
				.build();

		/*
        INDArray syn = Nd4j.create(new int[]{arrays.size(), arrays.get(0).columns()});
        for (int i = 0; i < syn.rows(); i++) {
            syn.putRow(i,arrays.get(i));
        }
		 */
		INDArray syn = Nd4j.vstack(arrays);

		Nd4j.clearNans(syn);
		lookupTable.setSyn0(syn);

		return fromPair(Pair.makePair((InMemoryLookupTable) lookupTable, (VocabCache) cache));
	}
	@SuppressWarnings("rawtypes")
	public WordVectors fromPair(Pair<InMemoryLookupTable, VocabCache> pair)
	{
		WordVectorsImpl vectors = new WordVectorsImpl();
		vectors.setLookupTable(pair.getFirst());
		vectors.setVocab(pair.getSecond());
		vectors.setModelUtils(new BasicModelUtils());
		return vectors;
	}
	
	
}