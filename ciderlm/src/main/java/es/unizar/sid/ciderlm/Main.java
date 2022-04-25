package es.unizar.sid.ciderlm;

import de.uni_mannheim.informatik.dws.melt.matching_data.TestCase;
import de.uni_mannheim.informatik.dws.melt.matching_data.Track;
import de.uni_mannheim.informatik.dws.melt.matching_data.TrackRepository;
import de.uni_mannheim.informatik.dws.melt.matching_eval.ExecutionResultSet;
import de.uni_mannheim.informatik.dws.melt.matching_eval.Executor;
import de.uni_mannheim.informatik.dws.melt.matching_eval.evaluator.EvaluatorCSV;
import de.uni_mannheim.informatik.dws.melt.matching_jena.TextExtractor;
import de.uni_mannheim.informatik.dws.melt.matching_jena_matchers.util.textExtractors.TextExtractorForTransformers;
import de.uni_mannheim.informatik.dws.melt.matching_jena_matchers.util.textExtractors.TextExtractorSet;
import de.uni_mannheim.informatik.dws.melt.matching_jena_matchers.util.textExtractors.TextExtractorShortAndLongTexts;
// import de.uni_mannheim.informatik.dws.melt.matching_ml.python.nlptransformers.SentenceTransformersMatcher;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Main {

	private static final Logger LOGGER = LoggerFactory.getLogger(Main.class);

	private static List<TextExtractor> extractorList = new ArrayList<>();
	private static List<Track> tracks = new ArrayList<>();

	// Transformer model options
	private static String transformerModels[] = { "sentence-transformers/all-MiniLM-L6-v2" };
	// "sentence-transformers/distiluse-base-multilingual-cased-v2" }; //
	// https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2
	// transformerModels.add(
	// "sentence-transformers/distiluse-base-multilingual-cased-v1");
	// transformerModels.add(
	// "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2");
	// transformerModels.add(
	// "sentence-transformers/paraphrase-multilingual-mpnet-base-v2");

	public static void main(String[] args) {
		LOGGER.info("---- CIDERLM ----");

		// Extractor options
		extractorList.add(new TextExtractorSet());
		extractorList.add(new TextExtractorShortAndLongTexts());
		extractorList.add(new TextExtractorForTransformers());

		// Tracks options: https://dwslab.github.io/melt/track-repository
		// Languages in Multifarm: ar, cn, cz, de, en, es, fr, nl, pt, ru
		/// tracks.add(TrackRepository.Multifarm.ALL_IN_ONE_TRACK);
		tracks.add(TrackRepository.Multifarm.getSpecificMultifarmTrack("en-es"));

		// Using TextExtractorForTransformers
		TextExtractor textExtractor = extractorList.get(0);

		// Remove transformer tokenizer parellism warning
		/*
		 * Map<String, String> env = System.getenv();
		 * Map<String, String> writableEnv = (Map<String, String>) env;
		 * writableEnv.put("TOKENIZERS_PARALLELISM", "false");
		 */

		zeroShotEvaluation(transformerModels, tracks, textExtractor);

	}

	static void zeroShotEvaluation(String[] transformerModels, List<Track> tracks, TextExtractor textExtractor) {
		LOGGER.info("Mode: ZEROSHOT");
		List<TestCase> testCases = new ArrayList<>();
		for (Track track : tracks)
			testCases.addAll(track.getTestCases());

		ExecutionResultSet ers = new ExecutionResultSet();

		for (String transformerModel : transformerModels) {

			LOGGER.info("Processing transformer model: " + transformerModel);
			String configurationName = "zero_" + transformerModel + "_" +
					textExtractor.getClass().getSimpleName();

			try {
				if (testCases.size() > 0) {

					boolean isAutoThresholding = false;
					boolean isMultipleTextsToMultipleExamples = false;
					File transformersCache = null;
					String gpu = "";

					/*
					 * ers.addAll(Executor.run(testCases, new ApplyModelPipelineTransformerChanged(
					 * transformerModel, new RecallMatcherGenericTransformerChanged(20, true),
					 * textExtractor,
					 * isAutoThresholding), configurationName));
					 */
					ers.addAll(Executor.run(testCases, new ApplyModelPipelineTransformer(gpu,
							transformerModel, transformersCache, new RecallMatcherAll2All(),
							isMultipleTextsToMultipleExamples, textExtractor, isAutoThresholding), configurationName));
				}
			} catch (Exception e) {
				LOGGER.warn("A problem occurred with transformer: '{}'.\n"
						+ "Continuing process...",
						transformerModel, e);
			}
		}
		EvaluatorCSV evaluator = new EvaluatorCSV(ers);
		evaluator.writeToDirectory();
	}

}
