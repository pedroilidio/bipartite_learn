"""Drug-protein interaction prediction datasets from Yamanishi et al., 2008.

The original files can be downloaded from:
    http://web.kuicr.kyoto-u.ac.jp/supp/yoshi/drugtarget/

This module provides utilities for fetching and loading the four gold-standard
datasets provided by [1]_. Each dataset is comprised of two feature matrices
describing a set of drugs and a set of target proteins, as well as a binary
interaction matrix determining the interaction between a drug and a protein is
confirmed (value 1) or that no interaction is known to exist between them (value
0).

The four datasets correspond to four protein families: enzymes, ion channels,
GPCRs and nuclear receptors. The input features are similarity matrices between
the instances on each axis. The score of a Smith-Waterman pairwise alignment is
taken as the the similarity between proteins, whereas the SIMCOMP score is used
for the similarity between drug molecules.

References
----------
.. [1] Yoshihiro Yamanishi, Michihiro Araki, Alex Gutteridge, Wataru Honda,
Minoru Kanehisa, Prediction of drug–target interaction networks from the
integration of chemical and genomic spaces, Bioinformatics, Volume 24, Issue 13,
July 2008, Pages i232–i240, https://doi.org/10.1093/bioinformatics/btn162
"""
# Author: Pedro Ilídio <ilidio@alumni.usp.br>
# License: BSD 3 clause

from pathlib import Path
from sklearn.utils import check_pandas_support, check_symmetric
from .loader import get_data_home, BaseRemoteFileLoader, BipartiteDatasetLoader

__all__ = [
    "EnzymesLoader",
    "IonChannelsLoader",
    "GPCRsLoader",
    "NuclearReceptorsLoader",
]

# The original data can be found here.
BASE_URL = "http://web.kuicr.kyoto-u.ac.jp/supp/yoshi/drugtarget/"


class YamanishiFileLoader(BaseRemoteFileLoader):
    def load_local(self, as_frame=False):
        check_pandas_support(f"{type(self).__name__}.load_local()")
        import pandas as pd
        data = pd.read_table(self.local_path, index_col=0)
        if as_frame:
            return data
        return data.values


class EnzymesLoader(BipartiteDatasetLoader):
    """Binary interaction prediction between enzymes and drug molecules.

    This is one of four gold-standand datasets for drug-protein interaction
    prediction introduced by Yamanishi et al., 2008 [1]_.
    
    The input features are similarity matrices among the instances on each
    axis and the target matrix is a binary interaction matrix determining the
    the existence of an experimentally validated interaction (value 1) or the
    absence of information about an interaction (value 0).

    The score of a Smith-Waterman pairwise alignment is taken as the the
    similarity between proteins, whereas the SIMCOMP score is used for the
    similarity between drug molecules.

    The original files can be downloaded from:
        http://web.kuicr.kyoto-u.ac.jp/supp/yoshi/drugtarget/

    References
    ----------
    .. [1] Yoshihiro Yamanishi, Michihiro Araki, Alex Gutteridge, Wataru Honda,
    Minoru Kanehisa, Prediction of drug–target interaction networks from the
    integration of chemical and genomic spaces, Bioinformatics, Volume 24, Issue
    13, July 2008, Pages i232–i240,
    https://doi.org/10.1093/bioinformatics/btn162
    """

    def __init__(self, base_dir: str | Path | None = None):
        """Initialize the loader.
        Parameters
        ----------
        base_dir : str | Path | None, default=None
            The path to the directory where the data will be stored. If None,
            the data will be stored in the directory returned by
            :func:`bipartite_learn.datasets.get_data_home`.
        """
        super().__init__(
            filepath="dpi_enzymes",
            base_dir=base_dir or get_data_home(),
            y_loader=YamanishiFileLoader(
                url=BASE_URL + "e_admat_dgc.txt",
                checksum=(
                    "bf73f8f0c1a71f9fe92d1e0d8a1065a1122b383532cd47fc73df1cb0d"
                    "9ba6c10"
                ),
            ),
            X_loader=[
                YamanishiFileLoader(
                    url=BASE_URL + "e_simmat_dg.txt",
                    checksum=(
                        "14816c2bba2193d62f52ab53bd51fe9a490b28e98cf33df392a5d"
                        "a2bf24e5607"
                    ),
                ),
                YamanishiFileLoader(
                    url=BASE_URL + "e_simmat_dc.txt",
                    checksum=(
                        "28105b9570c91b4eac688eaf92db2e4aacb7117bc8665cab527fc"
                        "5d45cd83606"
                    ),
                ),
            ],
        )


class IonChannelsLoader(BipartiteDatasetLoader):
    """Binary interactions between ion channels and drug molecules.

    This is one of four gold-standand datasets for drug-protein interaction
    prediction introduced by Yamanishi et al., 2008 [1]_.
    
    The input features are similarity matrices among the instances on each
    axis and the target matrix is a binary interaction matrix determining the
    the existence of an experimentally validated interaction (value 1) or the
    absence of information about an interaction (value 0).

    The score of a Smith-Waterman pairwise alignment is taken as the the
    similarity between proteins, whereas the SIMCOMP score is used for the
    similarity between drug molecules.

    The original files can be downloaded from:
        http://web.kuicr.kyoto-u.ac.jp/supp/yoshi/drugtarget/

    References
    ----------
    .. [1] Yoshihiro Yamanishi, Michihiro Araki, Alex Gutteridge, Wataru Honda,
    Minoru Kanehisa, Prediction of drug–target interaction networks from the
    integration of chemical and genomic spaces, Bioinformatics, Volume 24, Issue
    13, July 2008, Pages i232–i240,
    https://doi.org/10.1093/bioinformatics/btn162
    """

    def __init__(self, base_dir: str | Path | None = None):
        """Initialize the loader.
        Parameters
        ----------
        base_dir : str | Path | None, default=None
            The path to the directory where the data will be stored. If None,
            the data will be stored in the directory returned by
            :func:`bipartite_learn.datasets.get_data_home`.
        """
        super().__init__(
            filepath="dpi_ion_channels",
            base_dir=base_dir or get_data_home(),
            y_loader=YamanishiFileLoader(
                url=BASE_URL + "ic_admat_dgc.txt",
                checksum=(
                    "ac9df3498bc12af5174503e5f1e09ce26a6d78d753861a5eca55425f9"
                    "f9ae2c9"
                ),
            ),
            X_loader=[
                YamanishiFileLoader(
                    url=BASE_URL + "ic_simmat_dg.txt",
                    checksum=(
                        "e15626145623124ad42a45412c544d5fed5e079003727df784d29"
                        "ed3ca72efef"
                    ),
                ),
                YamanishiFileLoader(
                    url=BASE_URL + "ic_simmat_dc.txt",
                    checksum=(
                        "c1078ca88528d75ddb43d4ed2559391d7d8f5375ec572e6e9112e"
                        "519d9cdbbf9"
                    ),
                ),
            ],
        )


class GPCRLoader(BipartiteDatasetLoader):
    """Binary interactions between G-protein coupled receptors and drug molecules.

    This is one of four gold-standand datasets for drug-protein interaction
    prediction introduced by Yamanishi et al., 2008 [1]_.
    
    The input features are similarity matrices among the instances on each
    axis and the target matrix is a binary interaction matrix determining the
    the existence of an experimentally validated interaction (value 1) or the
    absence of information about an interaction (value 0).

    The score of a Smith-Waterman pairwise alignment is taken as the the
    similarity between proteins, whereas the SIMCOMP score is used for the
    similarity between drug molecules.

    The original files can be downloaded from:
        http://web.kuicr.kyoto-u.ac.jp/supp/yoshi/drugtarget/

    References
    ----------
    .. [1] Yoshihiro Yamanishi, Michihiro Araki, Alex Gutteridge, Wataru Honda,
    Minoru Kanehisa, Prediction of drug–target interaction networks from the
    integration of chemical and genomic spaces, Bioinformatics, Volume 24, Issue
    13, July 2008, Pages i232–i240,
    https://doi.org/10.1093/bioinformatics/btn162
    """

    def __init__(self, base_dir: str | Path | None = None):
        """Initialize the loader.
        Parameters
        ----------
        base_dir : str | Path | None, default=None
            The path to the directory where the data will be stored. If None,
            the data will be stored in the directory returned by
            :func:`bipartite_learn.datasets.get_data_home`.
        """
        super().__init__(
            filepath="dpi_gpcr",
            base_dir=base_dir or get_data_home(),
            y_loader=YamanishiFileLoader(
                url=BASE_URL + "gpcr_admat_dgc.txt",
                checksum=(
                    "49b2a8a0139dac3116399618bfefc9d1afcc276ce31b7de4e241d1b6e"
                    "62ef8ab"
                ),
            ),
            X_loader=[
                YamanishiFileLoader(
                    url=BASE_URL + "gpcr_simmat_dg.txt",
                    checksum=(
                        "c58c4311d2086a6914208ed0914598528ce54ade88eb52b0f737e"
                        "edcaf5001e4"
                    ),
                ),
                YamanishiFileLoader(
                    url=BASE_URL + "gpcr_simmat_dc.txt",
                    checksum=(
                        "ba9c1cf096f0f688d1341fec22ebdecb63162005394d3b5ec3e2f"
                        "2ef3fd40df9"
                    ),
                ),
            ],
        )


class NuclearReceptorsLoader(BipartiteDatasetLoader):
    """Binary interactions between nuclear receptors and drug molecules.

    This is one of four gold-standand datasets for drug-protein interaction
    prediction introduced by Yamanishi et al., 2008 [1]_.
    
    The input features are similarity matrices among the instances on each
    axis and the target matrix is a binary interaction matrix determining the
    the existence of an experimentally validated interaction (value 1) or the
    absence of information about an interaction (value 0).

    The score of a Smith-Waterman pairwise alignment is taken as the the
    similarity between proteins, whereas the SIMCOMP score is used for the
    similarity between drug molecules.

    The original files can be downloaded from:
        http://web.kuicr.kyoto-u.ac.jp/supp/yoshi/drugtarget/

    References
    ----------
    .. [1] Yoshihiro Yamanishi, Michihiro Araki, Alex Gutteridge, Wataru Honda,
    Minoru Kanehisa, Prediction of drug–target interaction networks from the
    integration of chemical and genomic spaces, Bioinformatics, Volume 24, Issue
    13, July 2008, Pages i232–i240,
    https://doi.org/10.1093/bioinformatics/btn162
    """

    def __init__(self, base_dir: str | Path | None = None):
        """Initialize the loader.
        Parameters
        ----------
        base_dir : str | Path | None, default=None
            The path to the directory where the data will be stored. If None,
            the data will be stored in the directory returned by
            :func:`bipartite_learn.datasets.get_data_home`.
        """
        super().__init__(
            filepath="dpi_nuclear_receptors",
            base_dir=base_dir or get_data_home(),
            y_loader=YamanishiFileLoader(
                url=BASE_URL + "nr_admat_dgc.txt",
                checksum=(
                    "74256ec88b415dbdb93735e93797955e82587f3154988e99787a4e604"
                    "111963b"
                ),
            ),
            X_loader=[
                YamanishiFileLoader(
                    url=BASE_URL + "nr_simmat_dg.txt",
                    checksum=(
                        "5bf651f310b037320fe5d8aa61dfa15e0d8af4b0fd50c39692010"
                        "85e64b631e1"
                    ),
                ),
                YamanishiFileLoader(
                    url=BASE_URL + "nr_simmat_dc.txt",
                    checksum=(
                        "f80e101f284934b85c5dc7601ecaca2cc86c62a946141b2587f3a"
                        "c8482c9a7c1"
                    ),
                ),
            ],
        )