package es.unizar.sid.ciderlm;

import de.uni_mannheim.informatik.dws.melt.matching_data.GoldStandardCompleteness;
import de.uni_mannheim.informatik.dws.melt.matching_eval.paramtuning.ConfidenceFinder;
import de.uni_mannheim.informatik.dws.melt.matching_jena.MatcherYAAAJena;
import de.uni_mannheim.informatik.dws.melt.matching_jena.TextExtractor;
import de.uni_mannheim.informatik.dws.melt.matching_jena_matchers.filter.ConfidenceFilter;
import de.uni_mannheim.informatik.dws.melt.matching_jena_matchers.filter.extraction.MaxWeightBipartiteExtractor;
import de.uni_mannheim.informatik.dws.melt.matching_jena_matchers.metalevel.ConfidenceCombiner;
import de.uni_mannheim.informatik.dws.melt.matching_jena_matchers.util.StringProcessing;
import de.uni_mannheim.informatik.dws.melt.matching_ml.python.nlptransformers.TransformersFilter;
import de.uni_mannheim.informatik.dws.melt.yet_another_alignment_api.Alignment;
import java.io.File;
import java.util.Properties;
import org.apache.jena.ontology.OntModel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * This pipeline matcher applies the given model.
 */
public class ApplyModelPipeline extends MatcherYAAAJena {

    private static final Logger LOGGER = LoggerFactory.getLogger(ApplyModelPipeline.class);

    private final MatcherYAAAJena recallMatcher;
    private final TransformersFilter transformersFilter;

    public ApplyModelPipeline(String transformerModel, MatcherYAAAJena recallMatcher, TextExtractor te) {
        TextExtractor textExtractor = te;
        textExtractor = TextExtractor.appendStringPostProcessing(textExtractor,
                StringProcessing::normalizeOnlyCamelCaseAndUnderscore);
        this.transformersFilter = new TransformersFilter(textExtractor, transformerModel);
        this.recallMatcher = recallMatcher;
    }

    @Override
    public Alignment match(OntModel source, OntModel target, Alignment inputAlignment, Properties properties)
            throws Exception {

        Alignment recallAlignment = this.recallMatcher.match(source, target, new Alignment(), properties);
        LOGGER.info("Recall alignment with {} correspondences", recallAlignment.size());

        /* if (recallAlignment.size() > 50_000) {
            LOGGER.info("Optimizing the transformers filter, since the recall alignment is very large.");
            this.transformersFilter.setOptimizeAll(true);
        } */

        Alignment alignmentWithConfidence = this.transformersFilter.match(source, target, recallAlignment, properties);
        
        /* if (recallAlignment.size() > 50_000) {
            // set to false for next time
            this.transformersFilter.setOptimizeAll(false);
        } */

        // now we need to set the transformer confidence as main confidence for the MWB
        // extractor
        ConfidenceCombiner confidenceCombiner = new ConfidenceCombiner(TransformersFilter.class);
        Alignment alignmentWithOneConfidence = confidenceCombiner.combine(alignmentWithConfidence);

        // just for logging
        double bestConfidenceF1 = ConfidenceFinder.getBestConfidenceForFmeasure(inputAlignment,
                alignmentWithOneConfidence,
                GoldStandardCompleteness.PARTIAL_SOURCE_COMPLETE_TARGET_COMPLETE);

        // just for logging
        double bestConfidencePrecision = ConfidenceFinder.getBestConfidenceForPrecision(inputAlignment,
                alignmentWithOneConfidence,
                GoldStandardCompleteness.PARTIAL_SOURCE_COMPLETE_TARGET_COMPLETE);

        // just for logging
        double bestConfidenceF05 = ConfidenceFinder.getBestConfidenceForFmeasureBeta(inputAlignment,
                alignmentWithOneConfidence,
                GoldStandardCompleteness.PARTIAL_SOURCE_COMPLETE_TARGET_COMPLETE, 0.5);

        // just for logging
        double bestConfidenceF15 = ConfidenceFinder.getBestConfidenceForFmeasureBeta(inputAlignment,
                alignmentWithOneConfidence,
                GoldStandardCompleteness.PARTIAL_SOURCE_COMPLETE_TARGET_COMPLETE, 1.5);

        LOGGER.info("Best confidence F1: {}\nBest confidence precision: {}\nBest confidence F05: {}\nBest confidence " +
                "F15: {}", bestConfidenceF1,
                bestConfidencePrecision, bestConfidenceF05, bestConfidenceF15);

        Alignment potentiallyThresholded = alignmentWithOneConfidence;

        // run the extractor
        MaxWeightBipartiteExtractor extractorMatcher = new MaxWeightBipartiteExtractor();
        return extractorMatcher.match(source, target, potentiallyThresholded, properties);
    }
}
