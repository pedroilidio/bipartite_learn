# Author: Pedro Ilidio
# License: BSD 3 clause
from __future__ import annotations
from abc import ABCMeta, abstractmethod
import copy
from os import environ
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse
from urllib.request import urlretrieve
import numpy as np
from sklearn.utils._param_validation import validate_params
from sklearn.datasets._base import _sha256

__all__ = [
    "get_data_home",
    "BaseFileLoader",
    "BaseRemoteFileLoader",
    "BipartiteDatasetLoader",
]

try:
    import pandas as pd
    Data = np.ndarray | pd.DataFrame | str
except ModuleNotFoundError:
    Data = np.ndarray | str


def get_data_home(data_home: str | Path | None = None) -> Path:
    """Return the path of the bipartite_learn data directory.  This folder is
    used by some large dataset loaders to avoid downloading the data several
    times.  By default the data directory is set to a folder named
    'bipartite_learn_data' in the user home folder. Alternatively, it can be
    set by the 'BIPARTITE_LEARN_DATA' environment variable or programmatically
    by giving an explicit folder path. The '~' symbol is expanded to the user
    home folder.  If the folder does not already exist, it is automatically
    created.
    Parameters
    ----------
    data_home : str, default=None
        The path to bipartite_learn data directory. If `None`, the default path
        is `~/bipartite_learn_data`.
    Returns
    -------
    data_home: Path
        The path to bipartite_learn data directory.
    """
    if data_home is None:
        data_home = environ.get(
            "BIPARTITE_LEARN_DATA", "~/bipartite_learn_data"
        )
    data_home = Path(data_home).expanduser()
    data_home.mkdir(parents=True, exist_ok=True)
    return data_home


class InvalidChecksumError(Exception):
    def __init__(self, path, actual=None, expected=None):
        self.path = path
        self.actual = actual
        self.expected = expected

    def __str__(self):
        msg = (
            f"{self.path}'s checksum differs from the expected value. File may"
            "be corrupted."
        )
        if self.expected is None or self.actual is None:
            return msg
        return msg + f"\n\tExpected: {self.expected}\n\tActual: {self.actual}"


class BaseFileLoader(metaclass=ABCMeta):
    """Abstract base class for file loader objects.
    Attributes
    ----------
    filepath : Path
        A Path object representing the location of the file to be loaded.
    description : str or BaseFileLoader
        A string describing the file being loaded or a BaseFileLoader instance 
        that provides additional details about the file.
    """

    def __init__(
        self,
        *,
        filepath: str | Path,
        description: str | BaseFileLoader = "",
    ):
        """Initialize the loader.
        Parameters
        ----------
        filepath : str or Path
            A Path object representing the location of the file to be loaded.
        description : str or BaseFileLoader, default=""
            A string describing the file being loaded or a BaseFileLoader
            instance that provides additional details about the file.
        """
        self.filepath: Path = Path(filepath)
        self.set_description(description)

    @property
    def local_path(self) -> Path:
        """The complete path to the local file."""
        return self.filepath.resolve()
    
    def set_description(self, description: str | BaseFileLoader) -> None:
        """Set the description of the file being loaded."""
        if not isinstance(description, (str, BaseFileLoader)):
            raise TypeError(
                f"{description=} must be either a string or a BaseFileLoader "
                "instance."
            )
        self.description = description

    def get_description(self) -> str:
        """Return the text description of the file being loaded."""
        if isinstance(self.description, BaseFileLoader):
            return self.description.load()
        return self.description
    
    @abstractmethod
    def load(self, as_frame: bool = False) -> Data:
        """Load the file and return its contents.
        Parameters
        ----------
        as_frame : bool, default=False
            Whether to return the data as a pandas DataFrame or not.
        Returns
        -------
        str or np.ndarray or pandas.DataFrame
            The contents of the file.
        """


class BaseRemoteFileLoader(BaseFileLoader, metaclass=ABCMeta):
    """Abstract base class for loaders of remote files.
    Attributes
    ----------
    url : str
        The URL address of the file to be fetched.
    filepath : Path
        A Path object representing the location of the file to be loaded,
        relative to ``self.base_dir``.
    description : str or BaseFileLoader
        A string describing the file being loaded or a BaseFileLoader instance
        that loads such string.
    base_dir : Path
        A Path object representing the root directory for ``self.filepath``.
    checksum : str or None
        The expected checksum of the file. If ``None``, integrity test is not
        performed after download.
    hash_function : Callable or None
        A function receiving the local filepath and returning its checksum.
        If ``None``, integrity test is not performed after download.
    """

    @validate_params(
        {
            "url": [str],
            "filepath": [str, Path, None],
            "description": [str, BaseFileLoader],
            "base_dir": [str, Path, None],
            "checksum": [str, None],
            "hash_function": [callable, None],
        },
        prefer_skip_nested_validation=True,
    )
    def __init__(
        self,
        *,
        url: str,
        filepath: str | Path | None = None,
        description: str | BaseRemoteFileLoader = "",
        base_dir: str | Path | None = None,
        checksum: str | None = None,
        hash_function: str | None = _sha256,
    ):
        """Initialize the loader.
        Parameters
        ----------
        url : str
            The URL address of the file to be fetched.
        filepath : str, Path or None, default=None
            A Path object representing the location of the file to be loaded,
            relative to ``self.base_dir``. If ``None``, the filename is inferred
            from the URL.
        description : str or BaseFileLoader, default=""
            A string describing the file being loaded or a BaseFileLoader instance
            that loads such string.
        base_dir : str, Path or None, default=None
            A Path object representing the root directory for ``self.filepath``.
            If ``None``, uses the current working directory.
        checksum : str or None, default=None
            The expected checksum of the file. If ``None``, integrity test is not
            performed after download.
        hash_function : Callable or None, default=sklearn.datasets._base._sha256
            A function receiving the local filepath and returning its checksum.
            If ``None``, integrity test is not performed after download.
        """
        self.url = url
        self.checksum = checksum
        self.hash_function = hash_function
        self.set_base_dir(base_dir)

        super().__init__(
            filepath=filepath or Path(urlparse(url).path).name,
            description=description,
        )

    @property
    def local_path(self) -> Path:
        """The complete path to the local file."""
        return (self.base_dir / self.filepath).resolve()
    
    def set_base_dir(self, base_dir: str | Path | None) -> None:
        """Set the root directory for ``self.filepath``."""
        self.base_dir = Path('.') if base_dir is None else Path(base_dir)

    def rebase_dir(self, new_base: str | Path) -> None:
        """Redirect the target download directory to be under new_base."""
        self.set_base_dir(Path(new_base) / self.base_dir)
   
    def set_description(self, description: str | BaseFileLoader) -> None:
        """Set the description of the file being loaded."""
        if isinstance(description, BaseRemoteFileLoader):
            # Copying avoid side effects when rebasing the directory
            description = copy.deepcopy(description)
            description.set_base_dir(self.base_dir)
        super().set_description(description)

    def download(self) -> Path:
        """Download the file to ``self.local_path`` and verify its integrity.
        """
        self.base_dir.mkdir(parents=True, exist_ok=True)
        urlretrieve(self.url, self.local_path)

        # If either checksum or hash_function was not provided, the integrity
        # test is not performed.
        if (self.checksum and self.hash_function) is not None:
            actual = self.hash_function(self.local_path)
            if actual != self.checksum:
                raise InvalidChecksumError(
                    self.local_path, actual=actual, expected=self.checksum
                )

        return self.local_path

    def clear_local(self) -> None:
        """Delete the local copy of the file."""
        self.local_path.unlink(missing_ok=True)
 
    def load(self, as_frame: bool = False) -> Data:
        """Load the local file if available, otherwise download it first.
        Parameters
        ----------
        as_frame : bool, default=False
            Whether to return the data as a pandas DataFrame or not.
        Returns
        -------
        str or np.ndarray or pandas.DataFrame
            The contents of the file.
        """
        if not self.local_path.exists():
            self.download()
        return self.load_local(as_frame)
    
    @abstractmethod
    def load_local(self, as_frame: bool = False) -> Data:
        """Load the local file and raise error if it is not available.
        Parameters
        ----------
        as_frame : bool, default=False
            Whether to return the data as a pandas DataFrame or not.
        Returns
        -------
        str or np.ndarray or pandas.DataFrame
            The contents of the file.
        """


class BipartiteDatasetLoader(BaseRemoteFileLoader):
    """Basic loader for bipartite datasets.

    This class groups data loaders for each file of a bipartite dataset.

    Attributes
    ----------
    X_loader : list[BaseRemoteFileLoader]
        List of remote file loaders for the feature matrices.
    y_loader : BaseRemoteFileLoader
        Remote file loader for the interaction matrix.
    filepath : Path
        Name for the local directory where the downloaded files will be stored.
        The directory will be created if it does not exist.
    base_dir : Path
        Base directory for the filepath. If ``None``, uses the current
        directory. 
    description : str or BaseRemoteFileLoader
        String description of the dataset or a file loader that loads it.
    """

    @validate_params(
        {
            "X_loader": [list],
            "y_loader": [BaseFileLoader],
            "filepath": [str, Path],
            "base_dir": [str, Path, None],
            "description": [str, BaseFileLoader],
        },
        prefer_skip_nested_validation=True,
    )
    def __init__(
        self,
        *,
        X_loader: list[BaseFileLoader],
        y_loader: BaseFileLoader,
        filepath: str | Path,
        base_dir: str | Path = None,
        description: str | BaseFileLoader = "",
    ):
        """Initializes a BipartiteDatasetLoader instance.
        Parameters
        ----------
        X_loader : list[BaseRemoteFileLoader]
            List of remote file loaders for the feature matrices.
        y_loader : BaseRemoteFileLoader
            Remote file loader for the interaction matrix.
        filepath : str or Path
            Name for a local directory where the downloaded files will be
            stored. The directory will be created if it does not exist.
        base_dir : str, Path or None, default=None
            Base directory for the filepath. If ``None``, uses the current
            directory. 
        description : str, BaseRemoteFileLoader, default=""
            String description of the dataset or a file loader that loads it.
        Raises
        ------
        NotADirectoryError
            If the local_path already exists and is not a directory.
        """
        # Deepcopy to avoid side effects when rebasing the directory
        self.y_loader = copy.deepcopy(y_loader)
        self.X_loader = [copy.deepcopy(X)  for X in X_loader]

        self.filepath = Path(filepath)
        self.set_base_dir(base_dir)

        if self.local_path.exists() and not self.local_path.is_dir():
            raise NotADirectoryError(f"{self.local_path=} is not a directory.")

        self.set_description(description)
    
    def set_base_dir(self, base_dir: str | Path | None) -> None:
        """Set the root directory for self.filepath and update loaders."""
        super().set_base_dir(base_dir)

        if isinstance(self.y_loader, BaseRemoteFileLoader):
            self.y_loader.set_base_dir(self.local_path)

        for X in self.X_loader:
            if isinstance(X, BaseRemoteFileLoader):
                X.set_base_dir(self.local_path)

    def download(self) -> Path:
        """Download the dataset files and verify their integrity."""
        if isinstance(self.y_loader, BaseRemoteFileLoader):
            self.y_loader.download()
        for X in self.X_loader:
            if isinstance(X, BaseRemoteFileLoader):
                X.download()
        return self.local_path
    
    def clear_local(self) -> None:
        """Delete the local copies of the files."""
        if isinstance(self.y_loader, BaseRemoteFileLoader):
            self.y_loader.clear_local()
        for X in self.X_loader:
            if isinstance(X, BaseRemoteFileLoader):
                X.clear_local()
        self.local_path.rmdir()
    
    def load_local(self, as_frame: bool = False) -> tuple[list[Data], Data]:
        """Load the local files and raise error if one is not available.
        Parameters
        ----------
        as_frame : bool, default=False
            Whether to return the data as a pandas DataFrames or numpy arrays.
        Returns
        -------
        X, y : tuple[list[Data], Data]
            Where Data is either pandas.DataFrame or numpy.ndarray.
            A tuple with the list of feature matrices for each axis and their
            corresponding interaction matrix.
        """
        if isinstance(self.y_loader, BaseRemoteFileLoader):
            y_data = self.y_loader.load_local(as_frame=as_frame) 
        else:
            y_data = self.y_loader.load(as_frame=as_frame)

        X_data = []
        for X in self.X_loader:
            if isinstance(X, BaseRemoteFileLoader):
                X_data.append(X.load_local(as_frame=as_frame))
            else:
                X_data.append(X.load(as_frame=as_frame))

        return X_data, y_data

    def load(self, as_frame: bool = False) -> tuple[list[Data], Data]:
        """Load the local files if available, otherwise download them first.
        Parameters
        ----------
        as_frame : bool, default=False
            Whether to return the data as a pandas DataFrames or numpy arrays.
        Returns
        -------
        X, y : tuple[list[Data], Data]
            Where Data is either pandas.DataFrame or numpy.ndarray.
            A tuple with the list of feature matrices for each axis and their
            corresponding interaction matrix.
        """
        return super().load(as_frame=as_frame)