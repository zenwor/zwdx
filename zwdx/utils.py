import os 
from typing import Union

def getenv(var: str, dft: str = None) -> Union[str, None]:
    return os.getenv(var, dft)