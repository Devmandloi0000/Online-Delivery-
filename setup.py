from setuptools import setup ,find_packages
from typing import List

HYPON_E_DOT = "-e ."

def get_requirements(file_path:str)->List[str]:
    requirements = []
    with open (file_path) as file_obj:
        requirements = file_obj.readlines()
        [req.replace('\n',"") for req in requirements]
        if HYPON_E_DOT in requirements:
            requirements.remove(HYPON_E_DOT)
    
    return requirements



setup(
    name = 'REGRESSORPROJECT',
    version = "0.0.1",
    author = 'dev',
    author_email = "devmandloi37@gmail.com",
    install_requires =get_requirements('requirements.txt'),
    packages = find_packages() 
)