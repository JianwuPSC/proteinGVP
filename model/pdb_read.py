import numpy as np
import biotite.structure
from biotite.structure.io import pdbx, pdb
from biotite.structure.residues import get_residues
from biotite.structure import filter_solvent
from biotite.structure import get_chains
from biotite.sequence import ProteinSequence
from typing import Sequence, Tuple, List

## 得到coords 坐标和序列，序列和坐标分为target和bind两种链，target 氨基酸组成序列，bind 序列单个间隔。
## from pdb and data_config( target chain, bind chain), get coords and seqence(diff seq + (-----))

def get_whole_structure(fpath):
    """
    Returns:
        biotite.structure.AtomArray
    """
    if fpath.endswith('cif'):
        with open(fpath) as fin:
            pdbxf = pdbx.PDBxFile.read(fin)
        structure = pdbx.get_structure(pdbxf, model=1)
    elif fpath.endswith('pdb'):
        with open(fpath) as fin:
            pdbf = pdb.PDBFile.read(fin)
        structure = pdb.get_structure(pdbf, model=1)
    issolvent = filter_solvent(structure)
    structure = structure[~issolvent]
    chains = get_chains(structure)
    print(f'Found {len(chains)} chains:', chains, '\n')
    if len(chains) == 0:
        raise ValueError('No chains found in the input file.')

    structure = structure[structure.hetero == False]
    return structure,chains
    
def load_structure(fpath, chain=None,bind_chains=None):
    """
    Returns:
        biotite.structure.AtomArray
    """
    structure,chains = get_whole_structure(fpath)
    assert chain in chains, ValueError('target chain {} not found in pdb file'.format(chain))
    structure_target = structure[structure.chain_id == chain]
    structure_binds = []
    if bind_chains is not None and bind_chains is not False:
        for bind_chain in bind_chains:
            assert bind_chain in chains, ValueError('bind chain {} not found in pdb file'.format(bind_chain))
            structure_bind = structure[structure.chain_id == bind_chain]
            structure_binds.append(structure_bind)
    
    return structure_target, structure_binds

def extract_coords_from_structure(structure):
    """
    Args:
        structure: An instance of biotite AtomArray
    Returns:
        Tuple (coords, seq)
            - coords is an L x 3 x 3 array for N, CA, C coordinates
            - seq is the extracted sequence
    """
    coords = get_atom_coords_residuewise(["N", "CA", "C", "CB"], structure)
    residue_identities = get_residues(structure)[1]
    seq = ''.join([ProteinSequence.convert_letter_3to1(r) for r in residue_identities if r in ProteinSequence._dict_3to1.keys()])

    return coords, seq


def load_coords(fpath, chain,bind_chains=None):
    """
    Args:
        fpath: filepath to either pdb or cif file
        chain: the chain id
    Returns:
        Tuple (coords, seq)
            - coords is an L x 3 x 3 array for N, CA, C coordinates
            - seq is the extracted sequence
    """
    structure1,structure_binds = load_structure(fpath, chain,bind_chains=bind_chains)
    coords,seq = extract_coords_from_structure(structure1)
    coords_binds = []
    seq_binds = []
    for structure_bind in structure_binds:
        coords_bind,seq_bind = extract_coords_from_structure(structure_bind)
        coords_binds.append(coords_bind)
        seq_binds.append(seq_bind)
    return coords,seq,coords_binds,seq_binds

def get_atom_coords_residuewise(atoms: List[str], struct: biotite.structure.AtomArray):
    """
    Example for atoms argument: ["N", "CA", "C"]
    """
    def filterfn(s, axis=None):
        filters = np.stack([s.atom_name == name for name in atoms], axis=1)
        sum = filters.sum(0)
        if not np.all(sum <= np.ones(filters.shape[1])):
            raise RuntimeError("structure has multiple atoms with same name")
        index = filters.argmax(0)
        coords = s[index].coord
        coords[sum == 0] = float("nan")
        return coords

    return biotite.structure.apply_residue_wise(struct, struct, filterfn)


def get_coords_seq(pdbfile,config,ifbindchain=True,ifbetac=False):
    chain = config['target_chain']
    bind_chains = config['bindding_chain']
    addition_chain = []
    if ifbindchain and bind_chains:
        addition_chain.extend(bind_chains)
    coords, wt_seq, coords_binds, seq_binds = load_coords(pdbfile, chain,bind_chains=addition_chain) #  N, CA, C [263,3,3]; seq （aa）
    assert coords.shape[0] == len(wt_seq)
    seqs = []
    seqs.append(wt_seq)
    seqs.extend(seq_binds)
    seq_pad = '-'*10
    seq_bind_pad = seq_pad.join(seqs)
    coord_out = coords
    for i in coords_binds:
        coord_out = coord_cat(coord_out,i)
    if not ifbetac:
        coord_out = coord_out[:,:3,:]

    return coord_out,seq_bind_pad

def coord_cat(coord1,coord2):
    coord_pad = np.zeros((10, 4, 3))
    coord_pad[:] = np.inf
    coords_binds_pad = []
    coords_binds_pad.append(coord1)
    coords_binds_pad.append(coord_pad)
    coords_binds_pad.append(coord2)
    coords_binds_pad = np.vstack(coords_binds_pad)
    return coords_binds_pad
