import os
import os.path as osp
import csv
import ast
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


DATA_ROOT = "data"
SENTENCES_DIR = osp.join(DATA_ROOT, 'flickr/flickr30k_entities/Sentences')


class VGDataset(Dataset):
    def __init__(self, data_root=DATA_ROOT, data_type="train"):
        self.data_root = data_root
        self.data_type = data_type
        self.image_dir = osp.join(self.data_root, "vg")

        self.image_paths = []
        self.bboxes = []
        self.phrases = []

        self.anno_csv_path = osp.join(
            self.data_root, "ds_csv_ann/vg_split/csv_dir", self.data_type + ".csv"
        )
        with open(self.anno_csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.image_paths.append(row["image_fpath"])
                self.bboxes.append(row["bbox"])
                # make consistent with flickr30k interface
                self.phrases.append([row["phrase"]])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = osp.join(self.image_dir, self.image_paths[idx])
        image = np.array(Image.open(img_path).convert("RGB"))
        bbox = ast.literal_eval(self.bboxes[idx])
        phrases = self.phrases[idx]
        data = {"image": image, "bbox": bbox, "phrases": phrases}
        return data


class FlickrDataset(Dataset):
    def __init__(self, data_root=DATA_ROOT, data_type="flickr30k/train", phrase_types=[], sentences_dir=SENTENCES_DIR):
        self.data_root = data_root
        self.data_type = data_type
        self.image_dir = osp.join(self.data_root, "flickr/flickr30k-images")
        # Precomputed edge box can be downloaded from
        # https://github.com/BryanPlummer/pl-clc
        self.edge_box_dir = osp.join(self.data_root, "flickr/edge_box")

        self.phrase_types = phrase_types
        if self.phrase_types:
            phrase_type_dict = get_phrase_type_dict(sentences_dir)

        self.image_paths = []
        self.bboxes = []
        self.phrases = []

        self.anno_csv_path = osp.join(
            self.data_root,
            "ds_csv_ann/{}/csv_dir/{}.csv".format(*data_type.split("/")),
        )
        with open(self.anno_csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["img_id"].endswith('.jpg'):
                    image_path = row["img_id"]
                else:
                    image_path = row["img_id"] + ".jpg"
                bbox = row["bbox"]
                queries = row["query"].strip()
                if queries.startswith('['):
                    queries = ast.literal_eval(queries)
                else:
                    queries = [queries]
                if self.phrase_types:
                    # if phrase_type_dict[queries[0]] not in self.phrase_types:
                    if queries[0] not in phrase_type_dict or phrase_type_dict[queries[0]] not in self.phrase_types:
                        continue
                self.bboxes.append(bbox)
                self.image_paths.append(image_path)
                self.phrases.append(queries)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = osp.join(self.image_dir, self.image_paths[idx])
        edge_box_path = osp.join(
            self.edge_box_dir, self.image_paths[idx].replace('jpg', 'txt'))
        edge_box = None
        if osp.isfile(edge_box_path):
            edge_box = np.loadtxt(edge_box_path)
        image = np.array(Image.open(img_path).convert("RGB"))
        bbox = ast.literal_eval(self.bboxes[idx])
        phrases = self.phrases[idx]
        data = {"image": image, "bbox": bbox,
                "phrases": phrases, "edge_box": edge_box}
        return data


def get_sentence_data(fn):
    """
    Parses a sentence file from the Flickr30K Entities dataset

    input:
      fn - full file path to the sentence file to parse

    output:
      a list of dictionaries for each sentence with the following fields:
          sentence - the original sentence
          phrases - a list of dictionaries for each phrase with the
                    following fields:
                      phrase - the text of the annotated phrase
                      first_word_index - the position of the first word of
                                         the phrase in the sentence
                      phrase_id - an identifier for this phrase
                      phrase_type - a list of the coarse categories this
                                    phrase belongs to

    """
    with open(fn, 'r') as f:
        sentences = f.read().split('\n')

    annotations = []
    for sentence in sentences:
        if not sentence:
            continue

        first_word = []
        phrases = []
        phrase_id = []
        phrase_type = []
        words = []
        current_phrase = []
        add_to_phrase = False
        for token in sentence.split():
            if add_to_phrase:
                if token[-1] == ']':
                    add_to_phrase = False
                    token = token[:-1]
                    current_phrase.append(token)
                    phrases.append(' '.join(current_phrase))
                    current_phrase = []
                else:
                    current_phrase.append(token)

                words.append(token)
            else:
                if token[0] == '[':
                    add_to_phrase = True
                    first_word.append(len(words))
                    parts = token.split('/')
                    phrase_id.append(parts[1][3:])
                    phrase_type.append(parts[2:])
                else:
                    words.append(token)

        sentence_data = {'sentence': ' '.join(words), 'phrases': []}
        for index, phrase, p_id, p_type in zip(first_word, phrases, phrase_id, phrase_type):
            sentence_data['phrases'].append({'first_word_index': index,
                                             'phrase': phrase,
                                             'phrase_id': p_id,
                                             'phrase_type': p_type})

        annotations.append(sentence_data)

    return annotations


def get_phrase_type_dict(dirname):
    # dirname is the directory to the sentence files
    paths = [osp.join(dirname, x)
             for x in os.listdir(dirname) if x.endswith('.txt')]
    res = {}  # map from phrase to phrase type
    for path in paths:
        data = get_sentence_data(path)
        for sentence in data:
            for phrase in sentence['phrases']:
                res[phrase['phrase']] = phrase['phrase_type'][0]
    return res
