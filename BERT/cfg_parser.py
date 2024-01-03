import json
import yaml
import itertools

def dict_parser(x):
    if type(x) is dict:
        children = [dict_parser(val) for val in x.values()]
        return [dict(zip(x.keys(), tup)) for tup in itertools.product(*children)]
    elif type(x) is list:
        return list(itertools.chain(*map(dict_parser, x)))
    else:
        return [x]

def parse(fname): 
    with open(fname) as f:
        extension = fname.split('.')[-1].lower()
        if extension == 'json':
            orig = json.loads(f.read())
        elif extension in ['yaml', 'yml']:
            orig = yaml.load(f, Loader = yaml.FullLoader)
        else:
            print("Config extension unknown:", extension)
            assert(False)

    ret = dict_parser(orig)
    for c in ret:
        if "name_fields" in c and "name" in c:
            for name_field in c["name_fields"]:
                c["name"] = f'{c["name"]}_{name_field}-{c[name_field]}'
            
    return ret, orig