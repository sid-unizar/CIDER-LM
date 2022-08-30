package es.unizar.sid.ciderlm;

import de.uni_mannheim.informatik.dws.melt.matching_base.external.docker.MatcherDockerFile;
import de.uni_mannheim.informatik.dws.melt.matching_data.TrackRepository;
import de.uni_mannheim.informatik.dws.melt.matching_eval.ExecutionResultSet;
import de.uni_mannheim.informatik.dws.melt.matching_eval.Executor;
import de.uni_mannheim.informatik.dws.melt.matching_eval.evaluator.EvaluatorCSV;

import java.io.File;

public class EvaluationDockerMatcher {

	public static void main(String[] args) throws Exception {
		File dockerFile = new File("target/ciderlm-1.0-web-latest.tar.gz");
		MatcherDockerFile dockerMatcher = new MatcherDockerFile("ciderlm-1.0-web", dockerFile);

		// running the matcher on any task
		ExecutionResultSet ers = Executor.run(TrackRepository.Multifarm.ALL_IN_ONE_TRACK/* .getFirstTestCase() */,
				dockerMatcher);

		Thread.sleep(20000); // just to be sure that all logs are written.
		dockerMatcher.logAllLinesFromContainer(); // this will output the log of the container

		// evaluating our system
		EvaluatorCSV evaluatorCSV = new EvaluatorCSV(ers);

		// we should close the docker matcher so that docker cab shut down the container
		dockerMatcher.close();

		// writing evaluation results to disk
		evaluatorCSV.writeToDirectory();
	}
}
