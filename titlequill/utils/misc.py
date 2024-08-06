from typing import Iterable, Dict, Any


def list_of_dict_to_dict_of_list(list_of_dicts: Iterable[Dict[str, Any]]) -> Dict[str, Iterable[Any]]:
    
    keys = next(iter(list_of_dicts)).keys()
    dict_of_lists = {key: [d[key] for d in list_of_dicts] for key in keys}
    
    return dict_of_lists