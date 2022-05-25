import logging
import os.path
import tarfile
import xml.etree.ElementTree as ET
from typing import List, Dict, Callable

from tqdm import tqdm

logger = logging.getLogger(__name__)


def parse_text_node(node) -> Dict:
    try:
        language = node.attrib['lang']
    except:
        language = 'en'
    content = []
    for text in node.itertext():
        text = text.strip()
        if text != '':
            content.append(text)
    return {
        'language': language,
        'text': content,
    }


def parse_xml_file(f, name: str = None) -> Dict:
    tree = ET.parse(f)
    root = tree.getroot()
    result = {'file': name}

    node = root.find('abstract')
    if node is not None:
        output = parse_text_node(node)
        result['abstract'] = output['text']
        result['lang_abstract'] = output['language']

    node = root.find('description')
    if node is not None:
        output = parse_text_node(node)
        result['description'] = output['text']
        result['lang_description'] = output['language']

    node = root.find('claims')
    if node is not None:
        output = parse_text_node(node)
        result['claims'] = output['text']
        result['lang_claims'] = output['language']

    return result


def parse_xml_structure(f, name: str = None) -> Dict:
    tree = ET.parse(f)
    root = tree.getroot()
    result = {'file': name}
    for child in root:
        result[child.tag] = True
    return result


def process_compressed_file(path: str, func: Callable) -> List:
    results = []
    with tarfile.open(path, 'r:gz') as f_tar:
        for tarinfo in tqdm(f_tar, desc=path, total=len(f_tar.getmembers())):
            file_path = tarinfo.name
            _, ext = os.path.splitext(file_path)
            if ext.lower() == '.xml':
                with f_tar.extractfile(tarinfo) as f:
                    try:
                        _, name = os.path.split(file_path)
                        result = func(f, name)
                        if result is not None:
                            results.append(result)
                    except:
                        print("Exception thrown when reading from", name)
                        continue
    return results


def process_folder(path: str, func: Callable) -> List:
    file_names = os.listdir(path)
    results = []
    for name in file_names:
        file_path = os.path.join(path, name)
        _, ext = os.path.splitext(file_path)
        if ext.lower() == '.tgz':
            results.extend(process_compressed_file(file_path, func))
    return results


def process_folder_directly(path, func) -> List:
    file_names = os.listdir(path)
    results = []
    for name in tqdm(file_names):
        file_path = os.path.join(path, name)
        _, ext = os.path.splitext(file_path)
        if ext.lower() == '.xml':
            try:
                with open(file_path) as one_f:
                    result = func(one_f, name)
                    if result is not None:
                        results.append(result)
            except:
                print('Exception thrown when reading from', file_path)
    return results
    
