from collections import defaultdict
import logging
import sys
# from rdflib import Graph, RDFS, URIRef
from owlready2 import get_ontology, IRIS, sync_reasoner
from AlignmentFormat import serialize_mapping_to_tmp_file
from sentence_transformers import SentenceTransformer, util
from torch.nn.functional import relu
# from transformers import AutoModel, AutoTokenizer  # TODO remove unnecesary deps
import numpy as np
from scipy.optimize import linear_sum_assignment


modelname_list = [
    # 0. finetuned 50-50 :
    "oaei-resources/models-pre/model_sentence-transformers_distiluse-base-multilingual-cased-v2_50-50",
    # 1. finetuned 10-90 :
    "oaei-resources/models-pre/model_sentence-transformers_distiluse-base-multilingual-cased-v2_10-90",
    # 2. dbmcv1 :
    "sentence-transformers/distiluse-base-multilingual-cased-v1",
    # 3. dbmcv2 (BEST) :
    "sentence-transformers/distiluse-base-multilingual-cased-v2",
    # 4. pmmb2 :
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    # 5. pmml2 :
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    # 6. pxrmv1 :
    "sentence-transformers/paraphrase-xlm-r-multilingual-v1",
    # 7. finetuned 50-50 2-3-train pattern_en-fr-es
    "oaei-resources/models/model_sentence-transformers_distiluse-base-multilingual-cased-v2_50-50_oaei_final_pattern_en-fr-es",
    # 8. finetuned 50-50 cmt-conference-iasted sequence (class + property)
    "oaei-resources/models/model_sentence-transformers_distiluse-base-multilingual-cased-v2_50-50_cmt-conference-iasted_sequence_oaei_final/",
    # 9. finetuned 50-50 cmt-conference-iasted pattern_en (class + property)
    "oaei-resources/models/model_sentence-transformers_distiluse-base-multilingual-cased-v2_50-50_cmt-conference-iasted_pattern_en_oaei_final",
    # 10. finetuned 50-50 all-v2 sequence
    "oaei-resources/models/model_sentence-transformers_distiluse-base-multilingual-cased-v2_50-50_all-v2_sequence_oaei_final",
    # 11. finetuned 50-50 all-v2 pattern_en
    "oaei-resources/models/model_sentence-transformers_distiluse-base-multilingual-cased-v2_50-50_all-v2_pattern_en_oaei_final",
]

# TODO modelname = "sentence-transformers/paraphrase-xlm-r-multilingual-v1" # pxrmv1
modelname = modelname_list[10]
reasoner = False
threshold = 0.5

# 0. Label
# 1. Verbalize classes (children, parents) and properties (domain, range) (with sequence)
# 2. Verbalize classes (children, parents) and properties (domain, range) (with pattern)
# 3. Verbalize classes (children, parents) and properties (domain, range) (with pattern en-fr-es)
verbalization_function = 0


def encode_sentence_transformer(embeddings_model, source_label_list, target_label_list):

    source_embeddings_list = embeddings_model.encode(
        source_label_list, show_progress_bar=False)
    target_embeddings_list = embeddings_model.encode(
        target_label_list, show_progress_bar=False)

    return source_embeddings_list, target_embeddings_list


def verbalize_label_sequence(init_label, list_neighbors_one, list_neighbors_two):
    verbalization = init_label
    for label in list_neighbors_one:
        if not init_label == label:
            verbalization = verbalization + ", " + label
    for label in list_neighbors_two:
        if not init_label == label:
            verbalization = verbalization + ", " + label

    return verbalization


def verbalize_label_pattern_class(init_label, children, parents, language):

    if verbalization_function == 2:
        pattern = " is a "
    elif verbalization_function == 3:
        if language == "en":
            pattern = " is a "
        elif language == "es":
            pattern = " es un "
        elif language == "fr":
            pattern = " est un "
        else:
            pattern = " is a "

    verbalization = init_label
    for label in children:
        if not init_label == label:
            verbalization = verbalization + ", " + label + pattern + init_label
    for label in parents:
        if not init_label == label:
            verbalization = verbalization + ", " + init_label + pattern + label

    return verbalization


def verbalize_label_pattern_property(init_label, domains, ranges, language):

    pattern_domain = " has domain "
    pattern_range = " has range "

    verbalization = init_label
    for label in domains:
        if not init_label == label:
            verbalization = verbalization + ", " + init_label + pattern_domain + label
    for label in ranges:
        if not init_label == label:
            verbalization = verbalization + ", " + init_label + pattern_range + label

    return verbalization


def verbalize_neighbors(onto, iri, init_label, language, isClass=False, isProperty=False):

    if isClass:
        children = []
        parents = []

        for parent in onto.get_parents_of(IRIS[iri]):
            try:
                if len(parent.label) > 0:
                    parents.append(parent.label[0])
            except AttributeError:
                continue

        for child in onto.get_children_of(IRIS[iri]):
            try:
                if len(child.label) > 0:
                    children.append(child.label[0])
            except AttributeError:
                continue

        list_neighbors_one = children
        list_neighbors_two = parents

    elif isProperty:
        domains = []
        ranges = []

        for domain in IRIS[iri].domain:
            try:
                if len(domain.label) > 0:
                    domains.append(domain.label[0])
            except AttributeError:
                continue

        for range in IRIS[iri].range:
            try:
                if len(range.label) > 0:
                    ranges.append(range.label[0])
            except AttributeError:
                continue

        list_neighbors_one = domains
        list_neighbors_two = ranges

    if verbalization_function == 1:
        verbalization = verbalize_label_sequence(
            init_label, list_neighbors_one, list_neighbors_two)
    elif verbalization_function == 2 or verbalization_function == 3:
        if isClass:
            verbalization = verbalize_label_pattern_class(
                init_label, list_neighbors_one, list_neighbors_two, language)
        elif isProperty:
            verbalization = verbalize_label_pattern_property(
                init_label, list_neighbors_one, list_neighbors_two, language)

    return verbalization


def get_iri_label_lists(onto, generator, isClass=False, isProperty=False):
    iri_list = []
    label_list = []

    for item in generator:
        if len(item.label) < 1:
            continue
        label = item.label[0]
        language = item.label[0].lang
        iri = item.iri

        iri_list.append(iri)

        if verbalization_function == 0:
            verbalization = label
        elif verbalization_function > 0:
            verbalization = verbalize_neighbors(
                onto, iri, label, language, isClass, isProperty)

        label_list.append(verbalization)

    return iri_list, label_list


def match_sentence_transformer(source_onto, target_onto, embeddings_model, tokenizer):

    alignment = []

    # Fill IRI-Label lists from both source and target ontologies

    source_class_iri_list, source_class_label_list = get_iri_label_lists(
        source_onto, source_onto.classes(), isClass=True)

    target_class_iri_list, target_class_label_list = get_iri_label_lists(
        target_onto, target_onto.classes(), isClass=True)

    source_properties_iri_list, source_properties_label_list = get_iri_label_lists(
        source_onto, source_onto.properties(), isProperty=True)

    target_properties_iri_list, target_properties_label_list = get_iri_label_lists(
        target_onto, target_onto.properties(), isProperty=True)

    # Combine labels
    source_iri_list = source_class_iri_list + source_properties_iri_list
    source_label_list = source_class_label_list + source_properties_label_list
    target_iri_list = target_class_iri_list + target_properties_iri_list
    target_label_list = target_class_label_list + target_properties_label_list

    # Obtain embeddings from transformer model
    # TORCH:
    '''
    source_embeddings_list, target_embeddings_list = encode_torch(
        embeddings_model, tokenizer, source_label_list, target_label_list)
    '''
    # SENTENCE-TRANSFORMER
    source_embeddings_list, target_embeddings_list = encode_sentence_transformer(
        embeddings_model, source_label_list, target_label_list)

    # Create IRI-Embedding lists
    source_iri_embedding_list = list(
        zip(source_iri_list, source_embeddings_list))
    target_iri_embedding_list = list(
        zip(target_iri_list, target_embeddings_list))

    # Use Cosine Similarity to compare sentence embeddings
    # TORCH:
    # cos = torch.nn.CosineSimilarity()
    # SENTENCE-TRANSFORMER
    cos = util.cos_sim

    # Iterate for every pair of source-target ontology instances
    for source_iri, source_sentence_embedding in source_iri_embedding_list:
        for target_iri, target_sentence_embedding in target_iri_embedding_list:
            # Caluculate the cosine similarity for the pair
            confidence = cos(
                source_sentence_embedding[None, :], target_sentence_embedding[None, :])
            # Cosine Similarity: (-1, +1)
            # Use ReLU to transform the result to (0,+1)
            # CosSim = -1: Labels are oposite meaning, in this case it is not useful information. With ReLU -1 -> 0.
            confidence = relu(confidence)
            confidence = confidence.unsqueeze(1).item()

            # Append potential match to alignment
            alignment.append((source_iri, target_iri, "=", confidence))

    # Alignment shape: [('http://one.de', 'http://two.de', '=', 1.0), ...]
    return alignment


def hungarian_algorithm(source_onto, target_onto, input_alignment):

    # Get indexes for IRIs
    source_iri_indexes, target_iri_indexes = defaultdict(int), defaultdict(int)
    source_iri, target_iri = [], []

    for i, item in enumerate(list(source_onto.classes())+list(source_onto.properties())):
        iri = item.iri
        source_iri_indexes[iri] = i
        source_iri.append(iri)

    for i, item in enumerate(list(target_onto.classes())+list(target_onto.properties())):
        iri = item.iri
        target_iri_indexes[iri] = i
        target_iri.append(iri)

    confindences_matrix = np.empty(
        (len(source_iri_indexes), len(target_iri_indexes)))
    for match in input_alignment:
        confindences_matrix[source_iri_indexes[match[0]],
                            target_iri_indexes[match[1]]] = match[3]

    source_ind_list, target_ind_list = linear_sum_assignment(
        confindences_matrix, maximize=True)
    alignment = [(source_iri[source_ind], target_iri[target_ind], "=", confindences_matrix[source_ind,  target_ind])
                 for source_ind, target_ind in zip(source_ind_list, target_ind_list)]

    return alignment


def replaceDateForDateTime(fileName):
    # Substitute date for dateTime to allow OWL 2.0 reasoning

    sDateShort = '&xsd;date'
    sDateTimeShort = '&xsd;dateTime'
    sDateLong = 'http://www.w3.org/2001/XMLSchema#date'
    sDateTimeLong = 'http://www.w3.org/2001/XMLSchema#dateTime'

    with open(fileName, "rt") as f:
        file_content = f.read().replace(
            sDateLong, sDateTimeLong).replace(sDateShort, sDateTimeShort)

    with open(fileName, "w") as f:
        f.write(file_content)


def replaceDateTimeForDate(fileName):
    # Substitute date for dateTime to allow OWL 2.0 reasoning

    sDateShort = '&xsd;date'
    sDateTimeShort = '&xsd;dateTime'
    sDateLong = 'http://www.w3.org/2001/XMLSchema#date'
    sDateTimeLong = 'http://www.w3.org/2001/XMLSchema#dateTime'

    with open(fileName, "rt") as f:
        file_content = f.read().replace(
            sDateTimeLong, sDateLong).replace(sDateTimeShort, sDateShort)

    with open(fileName, "w") as f:
        f.write(file_content)


def match(source_url, target_url, input_alignment_url):

    logging.info("Python matcher info: Match " +
                 source_url + " to " + target_url)

    if reasoner:
        # Substitute date for dateTime to allow OWL 2.0 reasoning
        replaceDateForDateTime(source_url[5:])
        replaceDateForDateTime(target_url[5:])

    # TORCH:
    '''
    embeddings_model = AutoModel.from_pretrained(modelname)
    tokenizer = AutoTokenizer.from_pretrained(modelname)
    '''
    # SENTENCE-TRANSFORMER
    embeddings_model = SentenceTransformer(modelname)
    tokenizer = None

    # Extract source-target ontologies from URL
    source_url = "file://" + source_url[5:]
    source_onto = get_ontology(source_url).load()
    logging.info("Read source with %s classes and %s properties.", len(
        list(source_onto.classes())), len(list(source_onto.properties())))
    target_url = "file://" + target_url[5:]
    target_onto = get_ontology(target_url).load()
    logging.info("Read target with %s classes and %s properties.", len(
        list(target_onto.classes())), len(list(target_onto.properties())))

    if reasoner:
        # Substitute date for dateTime to allow OWL 2.0 reasoning
        replaceDateTimeForDate(source_url[5:])
        replaceDateTimeForDate(target_url[5:])

    if reasoner:
        # Use reasoner to find other relations inside ontology
        with source_onto:
            sync_reasoner()
        logging.info("Read source with %s classes and %s properties after reasoning.", len(
            list(source_onto.classes())), len(list(source_onto.properties())))

        with target_onto:
            sync_reasoner()
        logging.info("Read target with %s classes and %s properties after reasoning.", len(
            list(target_onto.classes())), len(list(target_onto.properties())))

    # Obtain confindences using transformers model
    transformers_alignment = match_sentence_transformer(
        source_onto, target_onto, embeddings_model, tokenizer)

    intermidiate_alignment = transformers_alignment

    # Obtain complete alignment using hungarian algorithm
    hungarian_alignment = hungarian_algorithm(
        source_onto, target_onto, intermidiate_alignment)

    resulting_alignment = []
    # Remove matches with low confidence from alignment
    for match in hungarian_alignment:
        if match[3] > threshold:
            resulting_alignment.append(match)

    # Serialize final alignment to file and return it
    alignment_file_url = serialize_mapping_to_tmp_file(resulting_alignment)
    return alignment_file_url


def main(argv):
    if len(argv) == 2:
        print(match(argv[0], argv[1], None))
    elif len(argv) >= 3:
        if len(argv) > 3:
            logging.error("Too many parameters but we will ignore them.")
        print(match(argv[0], argv[1], argv[2]))
    else:
        logging.error(
            "Too few parameters. Need at least two (source and target URL of ontologies"
        )


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s:%(message)s", level=logging.INFO
    )
    main(sys.argv[1:])
