package es.unizar.sid.ciderlm;

import de.uni_mannheim.informatik.dws.melt.matching_data.TrackRepository;
import de.uni_mannheim.informatik.dws.melt.matching_data.LocalTrack;
import de.uni_mannheim.informatik.dws.melt.matching_eval.ExecutionResult;
import de.uni_mannheim.informatik.dws.melt.matching_eval.ExecutionResultSet;
import de.uni_mannheim.informatik.dws.melt.matching_eval.Executor;
import de.uni_mannheim.informatik.dws.melt.matching_eval.evaluator.EvaluatorCSV;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Please make sure that you have Python and the required dependencies
 * installed.
 */
public class EvaluateDemoPythonMatcher {

    private static final Logger LOGGER = LoggerFactory.getLogger(EvaluateDemoPythonMatcher.class);

    public static void main(String[] args) {

        ExecutionResultSet result = Executor.run(new LocalTrack("confOf-sigkdd", "1",
                "/home/javier/oaei_track_cache/oaei.webdatacommons.org/multifarm/confOf-sigkdd/"), new Main());

        // ExecutionResultSet result =
        // Executor.run(TrackRepository.Multifarm.ALL_IN_ONE_TRACK, new Main());

        // ExecutionResultSet result = Executor.run(TrackRepository.Conference.V1, new
        // Main()); // Different track

        ExecutionResult r = result.iterator().next();
        LOGGER.info("Python matcher run returned {} correspondences.", r.getSystemAlignment().size());
        EvaluatorCSV evaluatorCSV = new EvaluatorCSV(result);
        evaluatorCSV.writeToDirectory();
    }

}
