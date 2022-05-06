"""Convert image features from bottom up attention to numpy array"""

# Example
# python convert_data.py --imgid_list 'img_id.txt' --input_file 'test.csv' --output_file 'test.npy'

import os
import base64
import csv
import sys
import zlib
import json
import argparse

import numpy as np
import pickle as pkl

parser = argparse.ArgumentParser()
parser.add_argument('--imgid_list', default='data/coco_precomp/train_ids.txt',
                    help='Path to list of image id')
parser.add_argument('--input_file', default='/media/data/kualee/coco_bottom_up_feature/trainval_36/trainval_resnet101_faster_rcnn_genome_36.tsv',
                    help='tsv of all image data (output of bottom-up-attention/tools/generate_tsv.py), \
                    where each columns are: [image_id, image_w, image_h, num_boxes, boxes, features].')
parser.add_argument('--output_file', default='test.npy',
                    help='Output file path. the file saved in npy format')
parser.add_argument('--output_type', default='features', help='Which field to output, features or boxes')
parser.add_argument('--normalize_boxes', type=int, default=0, help='Whether to normalize bounding boxes to between 0 and 1')

opt = parser.parse_args()
print(opt)


meta = []
feature = {}
for line in open(opt.imgid_list):
    sid = int(line.strip())
    meta.append(sid)
    feature[sid] = None

csv.field_size_limit(sys.maxsize)
FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features', 'labels']

if __name__ == '__main__':
    with open(opt.input_file, "r+") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
        for item in reader:
            item['image_id'] = int(item['image_id'])
            item['image_h'] = int(item['image_h'])
            item['image_w'] = int(item['image_w'])
            item['num_boxes'] = int(item['num_boxes'])
            item['labels'] = item['labels']
            for field in ['boxes', 'features']:
                data = item[field]
                # buf = base64.decodestring(data)
                buf = base64.b64decode(data[1:])
                if field == 'features':
                    temp = np.frombuffer(buf, dtype=np.float32)
                else:
                    temp = np.frombuffer(buf, dtype=np.float64)
                item[field] = temp.reshape((item['num_boxes'],-1))
                if field == 'boxes' and opt.normalize_boxes:
                    scale = np.array([item['image_w'], item['image_h'], item['image_w'], item['image_h']])
                    scaled = item[field] / scale
                    item[field] = scaled
            if item['image_id'] in feature:
                if opt.output_type == 'features':
                    feature[item['image_id']] = item['features']
                elif opt.output_type == 'boxes':
                    feature[item['image_id']] = item['boxes']
                elif opt.output_type == 'labels':
                    feature[item['image_id']] = item['labels']
                else:
                    raise ValueError('Unrecognized output type')
    if opt.output_type == 'labels':
        data_out = [feature[sid] for sid in meta]
        pkl.dump(data_out, open(opt.output_file,'wb'))
    else:
        data_out = np.stack([feature[sid] for sid in meta], axis=0)
        # print (data_out)
        print("Final numpy array shape, min, max:", data_out.shape, np.min(data_out), np.max(data_out))
        np.save(opt.output_file, data_out)
