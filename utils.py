import os

import inspect
import argparse
import urllib.request

from tqdm import tqdm

# Argparser ----------------------------------------------------------------

def main(fn):
    signature = inspect.signature(fn)
    parser    = argparse.ArgumentParser(description = fn.__doc__)

    for name, arg in signature.parameters.items():
        argdef = [name]
        kwargdef = {}
        
        if arg.annotation is not inspect.Signature.empty:
            kwargdef["type"] = arg.annotation

        if arg.default is not inspect.Signature.empty:
            argdef[0]    = "--%s" % name
            kwargdef["default"] = arg.default

        parser.add_argument(*argdef, **kwargdef)
    
    args = parser.parse_args()
    return fn(**args.__dict__)


# Download helper ----------------------------------------------------------------

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download(remote_url, local_path):
    local_dir = os.path.dirname(local_path)
    if not os.path.exists(local_dir): os.makedirs(local_dir)

    # Test if url is reachable
    try:
        urllib.request.urlopen(remote_url)
    except urllib.error.HTTPError:
        raise ValueError(f'URL "{remote_url}" is not reachable')

    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc="Download %s" % remote_url.split("/")[-1]) as t:
        urllib.request.urlretrieve(remote_url,
                                    filename=local_path,
                                    reporthook=t.update_to)