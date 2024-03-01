import json
import os

def my_open(fpath,mode="w"):
   dirname=os.path.dirname(fpath)
   if len(dirname)>0 and not os.path.exists(dirname):
      os.makedirs(dirname)
   return open(fpath, mode)

def getTime(sec):
   m,s=divmod(sec,60)
   h,m=divmod(m,60)
   return "%d:%02d:%02d" % (h,m,s)

def human_format(num, digits=4, kilo = 1000):
   magnitude = 0
   while abs(num) >= kilo:
      magnitude += 1
      num /= kilo * 1.0
   return ('%.'+str(digits)+'g%s') % (num, ['', 'k', 'M', 'G', 'T', 'P'][magnitude])

def read_json_list(fpath):
    assert os.path.exists(
        fpath
    ), f"Provided file does not exist {fpath}"
    extension = os.path.splitext(fpath)[1].strip(".")
    if extension.lower() in ["json"]:
        with open(fpath, "r") as f:
            ret = json.load(f)
            assert isinstance(ret, list), "JSON content is not a list"
    else:
        raise ValueError("Error: unrecognized file extension '{extension}'!")
    return ret
