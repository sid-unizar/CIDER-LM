from collections import defaultdict
from email.policy import default
import logging
import sys
from rdflib import Graph,  RDFS
from AlignmentFormat import serialize_mapping_to_tmp_file
from transformers import AutoTokenizer, AutoModel
import torch
from hungarian_algorithm import algorithm
import numpy as np
from scipy.optimize import linear_sum_assignment


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):

    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    # Use mask to compute pooled embedding
    input_mask_expanded = attention_mask.unsqueeze(
        -1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    return sum_embeddings / sum_mask


def match_transformer(source_graph, target_graph, embeddings_model, tokenizer, input_alignment):

    alignment = []
    source_uri_list, source_label_list, target_uri_list, target_label_list = [], [], [], []

    # Fill URI-Label lists from both source and target ontologies
    for s, p, o in source_graph.triples((None, RDFS.label, None)):
        source_uri_list.append(str(s))
        source_label_list.append(str(o))
    for s, p, o in target_graph.triples((None, RDFS.label, None)):
        target_uri_list.append(str(s))
        target_label_list.append(str(o))

    # Combine label lists to tokenize jointly
    label_list = source_label_list + target_label_list
    encodings_list = tokenizer(
        label_list,
        add_special_tokens=True,
        padding=True,
        return_tensors="pt"
    )

    # Separate the encoding lists, accordingly with the label lists
    source_encodings_list = {'input_ids': encodings_list['input_ids'][:len(
        source_label_list)], 'attention_mask': encodings_list['attention_mask'][:len(source_label_list)]}
    target_encodings_list = {'input_ids': encodings_list['input_ids'][len(
        source_label_list):], 'attention_mask': encodings_list['attention_mask'][len(source_label_list):]}

    # Calculate token embeddings from tokenizer encodings and perform mean pooling to obtain a sentence embedding per label
    with torch.no_grad():
        source_embeddings_list = embeddings_model(**source_encodings_list)
        source_embeddings_list = mean_pooling(
            source_embeddings_list, source_encodings_list['attention_mask'])
        target_embeddings_list = embeddings_model(**target_encodings_list)
        target_embeddings_list = mean_pooling(
            target_embeddings_list, target_encodings_list['attention_mask'])

    # Create URI-Embedding lists
    source_uri_embedding_list = list(
        zip(source_uri_list, source_embeddings_list))
    target_uri_embedding_list = list(
        zip(target_uri_list, target_embeddings_list))

    # Use Cosine Similarity to compare sentence embeddings
    cos = torch.nn.CosineSimilarity()

    # Iterate for every pair of source-target ontology instances
    for source_uri, source_sentence_embedding in source_uri_embedding_list:
        for target_uri, target_sentence_embedding in target_uri_embedding_list:
            # Caluculate the cosine similarity for the pair
            confidence = cos(
                source_sentence_embedding[None, :], target_sentence_embedding[None, :])
            # Cosine Similarity: (-1, +1)
            # Use ReLU to transform the result to (0,+1)
            # CosSim = -1: Labels are oposite meaning, in this case it is not useful information. With ReLU -1 -> 0.
            confidence = torch.nn.functional.relu(confidence)
            confidence = confidence.unsqueeze(1).item()

            # if confidence > 0.5:
            # Append potential match to alignment
            alignment.append((source_uri, target_uri, "=", confidence))

    # Alignment shape: [('http://one.de', 'http://two.de', '=', 1.0), ...]
    return alignment


def hungarian_algorithm(source_graph, target_graph, input_alignment):

    # Get indexes for URIs
    source_uri_indexes, target_uri_indexes = defaultdict(int), defaultdict(int)
    source_uri, target_uri = [], []
    for i, (s, p, o) in enumerate(source_graph.triples((None, RDFS.label, None))):
        source_uri_indexes[str(s)] = i
        source_uri.append(str(s))
    for i, (s, p, o) in enumerate(target_graph.triples((None, RDFS.label, None))):
        target_uri_indexes[str(s)] = i
        target_uri.append(str(s))

    confindences_matrix = np.empty(
        (len(source_uri_indexes), len(target_uri_indexes)))
    for match in input_alignment:
        confindences_matrix[source_uri_indexes[match[0]],
                            target_uri_indexes[match[1]]] = match[3]

    source_ind_list, target_ind_list = linear_sum_assignment(
        confindences_matrix, maximize=True)
    alignment = [(source_uri[source_ind], target_uri[target_ind], "=", confindences_matrix[source_ind,  target_ind])
                 for source_ind, target_ind in zip(source_ind_list, target_ind_list)]

    ''' 
    bipartite_graph = defaultdict(lambda: defaultdict(None))
    for match in input_alignment:
        bipartite_graph[match[0]][match[1]] = match[3]
    for a in bipartite_graph:
        bipartite_graph[a] = dict(bipartite_graph[a])
    bipartite_graph = dict(bipartite_graph)

    # with open("sample.json", "w") as outfile:
    #     json.dump(bipartite_graph, outfile)
    # return []

    alignment = algorithm.find_matching(
        bipartite_graph, matching_type='max', return_type='list')
    alignment = [(match[0][0], match[0][1], "=", match[1])
                 for match in alignment] 
    '''

    return alignment


def match(source_url, target_url, input_alignment_url):

    logging.info("Python matcher info: Match " +
                 source_url + " to " + target_url)

    modelname = "sentence-transformers/distiluse-base-multilingual-cased-v2"
    embeddings_model = AutoModel.from_pretrained(modelname)
    tokenizer = AutoTokenizer.from_pretrained(modelname)

    # Extract source-target ontologies from URL
    source_graph = Graph()
    source_graph.parse(source_url)
    logging.info("Read source with %s triples.", len(source_graph))
    target_graph = Graph()
    target_graph.parse(target_url)
    logging.info("Read target with %s triples.", len(target_graph))

    # TODO input alignment
    input_alignment = None

    # Obtain confindences using transformers model
    transformers_alignment = match_transformer(
        source_graph, target_graph, embeddings_model, tokenizer, input_alignment)

    # TODO hungarian algorithm
    resulting_alignment = hungarian_algorithm(
        source_graph, target_graph, transformers_alignment)

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