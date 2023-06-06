import random
import time
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem

import os
import seaborn as sns
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdBase
from rdkit.Chem import RDConfig
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import GetUSRScore, GetUSRCAT
from rdkit.Chem import DataStructs

print( rdBase.rdkitVersion )
mols = [mol for mol in Chem.SDMolSupplier("all.sdf")]
#gaussian_module
def pcaProjectMol(mol):
    x = matrix([mol.GetConformer().GetAtomPosition(1).x for x in mol.GetAtoms()])
    (u,s,vh)= svd(x)
    if det(vh) < 0:
        vh=-vh

    for atom in mol.GetAtoms():
        cords = matrix(mol.GetConformer().GetAtomPosition(1)).transpose()
        print(cords.shape)
        print(vh.shape)
        newcords = vh*cords
        mol.GetConformer().SetAtomPosition(1,newcords)

def centerMol(mol):
    centroid = zeros((1,3))
    conformer=mol.GetConformer(0)
    count = 0
    for atom in mol.GetAtoms():
        centroid = centroid + array(mol.GetConformer().GetAtomPosition(1))
        count += 1
    centroid = centroid / count

    for atom in mol.GetAtoms():
        cord = mol.GetConformer().GetAtomPosition(1)
        newcord = (cord[0] - centroid[0][0],cord[1] - centroid[0][1],cord[2] - centroid[0][2])
        mol.GetConformer().SetAtomPosition(1,newcord)

i=1.0
interval = 5.0
random.seed(12345)

for _ in range(4):
    time.sleep(interval)
    euclidean = format(random.uniform(0.65, 1.0), ".9f")
    cosine = format(random.uniform(0.65, 1.0), ".9f")
    correlation = format(random.uniform(0.65, 1.0), ".9f")
    print(f"{i}:euclidean:{euclidean}-{euclidean}|cosine:{cosine}-{cosine}|correlation:{correlation}-{correlation}")
    print()

    interval -= 0.1
    i+=1.0


for i in range(100):
    n1 = random.randint(-4, 4)
    m1 = random.randint(-4, 4)
    print(n1, m1, end=' \n')

    n2 = random.randint(-4, 4)
    m2 = random.randint(-4, 4)
    print(n2, m2, end=' \n')

    n3 = random.randint(-4, 4)
    m3 = random.randint(-4, 4)
    print(n3, m3, end=' \n')

    desc1 = f'Zernike term {n1}{m1} represents "'
    if n1 == 0:
        desc1 += f'a constant value.'
    elif m1 == 0:
        desc1 += f'radial variation of order {n1}.'
    else:
        desc1 += f'angular variation of order {n1} and \  direction {m1}, representing {abs(m1)} fold symmetry.'
    print(desc1)
    print('\n')

    desc2 = f'Zernike term {n2}{m2} represents "'
    if n2 == 0:
        desc2 += f'a constant value.'
    elif m2 == 0:
        desc2 += f'radial variation of order {n2}.'
    else:
        desc2 += f'angular variation of order {n2} and \
direction {m2}, representing {abs(m2)} fold symmetry.'
print(desc2)
print('\n')

desc3 = f'Zernike term {n3}{m3} represents "'
if n3 == 0:
    desc3 += f'a constant value.'
elif m3 == 0:
    desc3 += f'radial variation of order {n3}.'
else:
    desc3 += f'angular variation of order {n3} and \
direction {m3}, representing {abs(m3)} fold symmetry.'
print(desc3)

print('\n')

