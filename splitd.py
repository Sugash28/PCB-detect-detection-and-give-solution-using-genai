import os
import xml.etree.ElementTree as ET

def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def convert_annotation(xml_file, output_dir):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    
    with open(os.path.join(output_dir, os.path.splitext(os.path.basename(xml_file))[0] + '.txt'), 'w') as out_file:
        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls not in class_mapping:
                continue
            cls_id = class_mapping[cls]
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
                 float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            bb = convert((w, h), b)
            out_file.write(f"{cls_id} {' '.join([str(a) for a in bb])}\n")

class_mapping = {
    'Missing Hole': 0,
    'Mouse Bite': 1,
    'Open Circuit': 2,
    'Short': 3,
    'Spur': 4,
    'Spurious Copper': 5
}

def convert_dataset(annotations_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for xml_file in os.listdir(annotations_dir):
        if xml_file.endswith(".xml"):
            convert_annotation(os.path.join(annotations_dir, xml_file), output_dir)

annotations_dir = "PCB_DATASET\Annotations"
output_dir = "PCB_DATASET/yolo_labels"
convert_dataset(annotations_dir, output_dir)
