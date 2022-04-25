package es.unizar.sid.ciderlm;

import de.uni_mannheim.informatik.dws.melt.matching_jena.MatcherYAAAJena;
import de.uni_mannheim.informatik.dws.melt.yet_another_alignment_api.Alignment;

import org.apache.jena.ontology.OntResource;
import org.apache.jena.util.iterator.ExtendedIterator;

import java.util.*;

import org.apache.jena.ontology.OntModel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class RecallMatcherAll2All extends MatcherYAAAJena {

	private static final Logger LOGGER = LoggerFactory.getLogger(RecallMatcherAll2All.class);

	@Override
	public Alignment match(OntModel source, OntModel target, Alignment alignment, Properties properties)
			throws Exception {

		if (alignment == null) {
			alignment = new Alignment();
		}

		LOGGER.debug("Match classes");

		matchResources(source.listClasses(), target.listClasses(), alignment);

		/*
		 * LOGGER.debug("Match properties");
		 * matchResources(source.listAllOntProperties(), target.listAllOntProperties(),
		 * alignment);
		 * 
		 * LOGGER.debug("Match instances");
		 * matchResources(source.listIndividuals(), target.listIndividuals(),
		 * alignment);
		 */

		LOGGER.debug("Finished");
		return alignment;
	}

	private void matchResources(ExtendedIterator<? extends OntResource> sourceResources,
			ExtendedIterator<? extends OntResource> targetResources, Alignment alignment) {

		List<? extends OntResource> sourceList = sourceResources.toList();
		List<? extends OntResource> targetList = targetResources.toList();

		// initial lists to get the whole ontology without the already aligned ones
		List<OntResource> sourcesToMatch = new ArrayList<>();
		List<OntResource> targetsToMatch = new ArrayList<>();

		if (alignment == null) {
			alignment = new Alignment();
		}

		for (OntResource r1 : sourceList) {
			String r1uri = r1.getURI();
			if (r1uri == null) {
				continue;
			}
			sourcesToMatch.add(r1);

		}

		for (OntResource r2 : targetList) {
			String r2uri = r2.getURI();
			if (r2uri == null) {
				continue;
			}

			targetsToMatch.add(r2);
		}

		for (OntResource r1 : sourcesToMatch) {
			// TODO remove
			String r1uri = r1.getURI();
			// String labelLeft = getLabel(r1);

			for (OntResource r2 : targetsToMatch) {

				// TODO remove
				String r2uri = r2.getURI();
				// String labelRight = getLabel(r2);

				alignment.add(r1uri, r2uri, 0.1);
			}
		}

	}

	public String getLabel(OntResource r) {
		if (r.getLabel(null) == null) {
			return r.getLocalName();
		} else {
			return r.getLabel(null);
		}
	}

}
