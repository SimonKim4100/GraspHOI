"""Convert the SMPL, MANO and SMPL+H downloads into plain NumPy pickles."""
import argparse
import os
import pickle

import numpy as np
from scipy import sparse

SMPL_OUT = os.path.join('data', 'models', 'SMPL', 'SMPL_NEUTRAL.pkl')
SMPLH_OUT = os.path.join('data', 'models', 'SMPL', 'SMPLH_NEUTRAL.pkl')
MANO_OUT = os.path.join('data', 'wilor_data', 'mano', 'MANO_RIGHT.pkl')

# numpy's array rebuild function
NDARRAY_RECONSTRUCT = np.empty(0).__reduce__()[0]


class ChLeaf:
    """Stand-in for chumpy.ch.Ch."""

    def __setstate__(self, state):
        self.x = state['x']

    def value(self):
        return np.asarray(self.x, dtype=np.float64)


class ChSelect:
    """Stand-in for chumpy.reordering.Select."""

    def __setstate__(self, state):
        self.a = state['a']
        self.idxs = state['idxs']
        self.preferred_shape = state['preferred_shape']

    def value(self):
        return self.a.value().ravel()[self.idxs].reshape(self.preferred_shape)


CHUMPY_STAND_INS = {('chumpy.ch', 'Ch'): ChLeaf, ('chumpy.reordering', 'Select'): ChSelect}


def python2_dtype(spec, align=False, copy=True):
    """np.dtype as called by Python 2 pickles."""
    return np.dtype(spec, bool(align), bool(copy))


class ChumpyFreeUnpickler(pickle.Unpickler):
    """Unpickler that maps chumpy classes to the stand-ins."""

    def find_class(self, module, name):
        if module.split('.')[0] == 'chumpy':
            if (module, name) not in CHUMPY_STAND_INS:
                raise pickle.UnpicklingError(f'no stand-in for the chumpy class {module}.{name}')
            return CHUMPY_STAND_INS[(module, name)]
        if (module, name) == ('scipy.sparse.csc', 'csc_matrix'):
            return sparse.csc_matrix
        if (module, name) == ('numpy.core.multiarray', '_reconstruct'):
            return NDARRAY_RECONSTRUCT
        if (module, name) == ('numpy', 'dtype'):
            return python2_dtype
        return super().find_class(module, name)


def load_plain_pickle(path):
    """Load a model pickle as a dict, chumpy objects as NumPy arrays."""
    with open(path, 'rb') as f:
        data = ChumpyFreeUnpickler(f, encoding='latin1').load()
    return {key: value.value() if isinstance(value, (ChLeaf, ChSelect)) else value for key, value in data.items()}


def merge_smplh(smplh_path, mano_left, mano_right):
    """SMPL+H arrays plus the PCA hand parameters of both MANO models."""
    with np.load(smplh_path) as npz:
        smplh = {key: npz[key] for key in npz.files}
    for name in ('hands_components', 'hands_coeffs', 'hands_mean'):
        smplh[name + 'l'] = mano_left[name]
        smplh[name + 'r'] = mano_right[name]
    return smplh


def write_pickle(data, path):
    """Write through a temporary file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + '.tmp', 'wb') as f:
        pickle.dump(data, f)
    os.replace(path + '.tmp', path)
    print(f'wrote {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--smpl', required=True, help='SMPL_NEUTRAL.pkl as downloaded')
    parser.add_argument('--mano-dir', required=True, help='folder holding MANO_LEFT.pkl and MANO_RIGHT.pkl')
    parser.add_argument('--smplh-dir', required=True, help='folder holding neutral/model.npz')
    parser.add_argument('--output-root', default='.', help='folder that receives data/ (default: current folder)')
    args = parser.parse_args()

    smpl = load_plain_pickle(args.smpl)
    mano_left = load_plain_pickle(os.path.join(args.mano_dir, 'MANO_LEFT.pkl'))
    mano_right = load_plain_pickle(os.path.join(args.mano_dir, 'MANO_RIGHT.pkl'))
    smplh = merge_smplh(os.path.join(args.smplh_dir, 'neutral', 'model.npz'), mano_left, mano_right)

    write_pickle(smpl, os.path.join(args.output_root, SMPL_OUT))
    write_pickle(smplh, os.path.join(args.output_root, SMPLH_OUT))
    write_pickle(mano_right, os.path.join(args.output_root, MANO_OUT))


if __name__ == '__main__':
    main()
