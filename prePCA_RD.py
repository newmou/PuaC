import sys
import math
from numpy import *
from numpy.linalg import svd,det
from rdkit import Chem
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
mols = [mol for mol in Chem.SDMolSupplier("aa2ar.sdf")]
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

def main():
    for f in sys.argv[1:]:
        mol=Chem.MolFromMolFile(f)

        centerMol(mol)
        pcaProjectMol(mol)

        molblock = Chem.MolToMolBlock(mol)
        print(molblock,file=open('f','w+'))
#usrcat_module
    for mol in mols:
        print("-----")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol,
                              useExpTorsionAnglePrefs=True,
                              useBasicKnowledge=True)
    usrcats = [GetUSRCAT(mol) for mol in mols]
    fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2) for mol in mols]

    data = {"tanimoto": [], "usrscore": []}

    for i in range(len(usrcats)):
        for j in range(i):
            tc = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            score = GetUSRScore(usrcats[i], usrcats[j])
            data["tanimoto"].append(tc)
            data["usrscore"].append(score)
            print(score, tc)
    df = pd.DataFrame(data)

    fig = sns.pairplot(df)
    fig.savefig('plot.png')
    
    desc = amp.descriptor.zernike(system, cut_off=5.0, n_max=4, device='GPU')
    
try:
    import psyco
    psyco.full()
except:
    pass

main()
