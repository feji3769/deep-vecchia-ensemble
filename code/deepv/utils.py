import pickle
import json
import os

def pickle_data(location, obj,  obj_name):
    """pickle data. 
    Args:
    location : (str) path to folder to save objects. 
    obj : list of objects to pickle. 
    obj_name: list of strings which name the objects to pickle. 
    """
    for obj_, obj_name_ in zip(obj, obj_name):
        filename = os.path.join(location, obj_name_ + ".pickle")
        with open(filename, 'wb') as handle:
            pickle.dump(obj_, handle)
            
def open_data(location, obj_name):
    """read pickled data back. 
    Args:
    location : (str) path to folder where objects live. 
    obj_name : list of strings which name the objects to restore.
    Returns:
    obj : obj : list of objects loaded. 
    """
    obj = []
    for obj_name_ in obj_name:
        filename = os.path.join(location, obj_name_ + ".pickle")
        with open(filename, 'rb') as handle:
            b = pickle.load(handle)
            obj.append(b)
    return obj


def json_dicts(location, obj, obj_name):
    """dump dictionaries to jsons. 
    Args:
    location : (str) path to dump dictionaries. 
    obj : list of dictionaries to dump.
    obj_name: list of strings which name the dictionaries.
    """
    for obj_, obj_name_ in zip(obj, obj_name):
        filename = location+ obj_name_ + ".json"
        with open(filename, 'w') as outfile:
            json.dump(obj_, outfile)


import json
def dict_from_json(location, jname):
    """loads dictionaries from json files. 
    Args:
    location : (str) path where dictionaries were dumped. 
    [jname] : list of strings which name the dictionaries. 
    Returns:
    obj : list of dictionaries loaded from jsons.
    """
    obj = []
    for jname_ in jname:
        filename = os.path.join(location, jname_ + ".json")
        with open(filename) as json_file:
            obj_ = json.load(json_file)
        obj.append(obj_)
    return obj