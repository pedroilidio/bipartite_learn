import pytest
import logging
from copy import deepcopy
from pathlib import Path
import numpy as np
from bipartite_learn import datasets
from bipartite_learn.datasets.base import (
    BaseFileLoader,
    BaseRemoteFileLoader,
    BipartiteDatasetLoader,
    InvalidChecksumError,
)


class BasicTextLoader(BaseFileLoader):
    def load(self, as_frame=False):
        with self.local_path.open() as f:
            return f.read()


class BasicRemoteTextLoader(BaseRemoteFileLoader):
    def download(self) -> Path:
        logging.info(f"Downloading {self.url} to {self.local_path}")
        return super().download()

    def load_local(self, as_frame=False):
        with self.local_path.open() as f:
            return f.read()


@pytest.fixture(scope="module")
def remote_url():
    return (
        "https://raw.githubusercontent.com/pedroilidio/bipartite_learn"
        "/main/README.md"
    )


@pytest.fixture(scope="module")
def checksum():
    return (
        "395fcfcdb437a918915f35bb7e6b1f7ef2a4dc10ef980340f1949ebfaac3112b"
    )


@pytest.fixture
def local_text_loader(tmp_path):
    out = tmp_path / "test.txt"
    with out.open("w") as f:
        f.write("test content")

    return BasicTextLoader(filepath=out)


@pytest.fixture
def remote_text_loader(tmp_path, remote_url):
    return BasicRemoteTextLoader(
        url=remote_url,
        base_dir=tmp_path,
    )


@pytest.fixture
def bipartite_dataset_loader(tmp_path, local_text_loader, remote_text_loader):
    return BipartiteDatasetLoader(
        X_loader=[local_text_loader, remote_text_loader],
        y_loader=remote_text_loader,
        base_dir=tmp_path,
        filepath="test_dataset",
    )


class TestLocalFileLoader:
    def test_local_path(self, tmp_path):
        loader = BasicTextLoader(filepath=tmp_path)
        assert loader.local_path == tmp_path.resolve()
    
    def test_set_description_str(self, local_text_loader):
        description = "test description"
        local_text_loader.set_description(description)
        assert local_text_loader.description == description
        assert local_text_loader.get_description() == description

    def test_set_description_loader(self, local_text_loader):
        local_text_loader.set_description(local_text_loader)
        assert local_text_loader.description is local_text_loader
        assert local_text_loader.get_description() == "test content"

    def test_set_description_remote_loader(
        self, local_text_loader, remote_text_loader,
    ):
        local_text_loader.set_description(remote_text_loader)
        assert local_text_loader.description is remote_text_loader
        assert (
            local_text_loader.get_description()
            == remote_text_loader.load_local()
        )

    def test_set_description_type_error(self, local_text_loader):
        with pytest.raises(TypeError):
            local_text_loader.set_description(123)
    

class TestRemoteFileLoader:
    def test_infer_filepath(self):
        url = (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/iris"
            "/iris.data?example=1"
        )
        loader = BasicRemoteTextLoader(url=url)
        assert loader.filepath == Path("iris.data")

    def test_set_description_loader(self, remote_text_loader, local_text_loader):
        remote_text_loader.set_description(local_text_loader)
        assert remote_text_loader.description is local_text_loader
        assert remote_text_loader.get_description() == "test content"

    def test_set_description_remote_loader(self, remote_text_loader):
        main_loader = deepcopy(remote_text_loader)
        main_loader.set_description(remote_text_loader)
        assert (
            main_loader.get_description() == remote_text_loader.load_local()
        )
        assert remote_text_loader.base_dir == main_loader.base_dir

    def test_set_base_dir(self, tmp_path, remote_text_loader):
        remote_text_loader.set_base_dir(tmp_path)
        assert remote_text_loader.base_dir == tmp_path

    def test_local_path(self, tmp_path, remote_text_loader):
        filepath = remote_text_loader.filepath
        remote_text_loader.set_base_dir(tmp_path)
        remote_text_loader.local_path == (tmp_path / filepath).resolve()

    def test_rebase_dir(self, tmp_path, remote_text_loader):
        loader = remote_text_loader
        filepath = remote_text_loader.filepath
        base_dir = tmp_path / "test/test1/test2"

        loader.set_base_dir("test2")
        loader.rebase_dir("test1")
        loader.rebase_dir("test")
        loader.rebase_dir(tmp_path)

        assert loader.base_dir == base_dir
        assert loader.local_path == base_dir / filepath
    
    def test_download(self, remote_text_loader):
        remote_text_loader.download()
        assert remote_text_loader.local_path.exists()

    def test_checksum_fail(self, remote_text_loader):
        remote_text_loader.checksum = "123"
        with pytest.raises(InvalidChecksumError):
            remote_text_loader.download()

    def test_correct_checksum(self, remote_text_loader, checksum):
        remote_text_loader.checksum = checksum
        remote_text_loader.download()
        assert remote_text_loader.local_path.exists()

    def test_clear_local(self, remote_text_loader):
        remote_text_loader.download()
        assert remote_text_loader.local_path.exists()
        remote_text_loader.clear_local()
        assert not remote_text_loader.local_path.exists()


class TestBipartiteDatasetLoader:
    def test_set_base_dir(self, bipartite_dataset_loader, tmp_path):
        loader = bipartite_dataset_loader
        local_path = tmp_path / loader.filepath 
        loader.set_base_dir(tmp_path)

        assert loader.base_dir == tmp_path
        assert loader.local_path == local_path
        assert loader.X_loader[1].base_dir == local_path
        assert loader.y_loader.base_dir == local_path

    def test_download(self, bipartite_dataset_loader, checksum):
        loader = bipartite_dataset_loader
        loader.X_loader[1].checksum = checksum
        loader.download()

        assert loader.X_loader[0].local_path.exists()
        assert loader.X_loader[1].local_path.exists()
        assert loader.y_loader.local_path.exists()

    def test_clear_local(self, bipartite_dataset_loader, checksum):
        loader = bipartite_dataset_loader
        loader.X_loader[1].checksum = checksum

        loader.download()
        assert loader.y_loader.local_path.exists()
        loader.clear_local()

        assert loader.X_loader[0].local_path.exists()  # Loader is local
        assert not loader.X_loader[1].local_path.exists()
        assert not loader.y_loader.local_path.exists()
        assert not loader.local_path.exists()
    
    def test_load(self, bipartite_dataset_loader):
        loader = bipartite_dataset_loader
        X, y = loader.load()

        assert X[0] == loader.X_loader[0].load()
        assert X[1] == loader.X_loader[1].load()
        assert y == loader.y_loader.load()


@pytest.mark.parametrize(
    "loader_class", [
        datasets.EnzymesLoader,
        datasets.IonChannelsLoader,
        datasets.GPCRLoader,
        datasets.NuclearReceptorsLoader,
    ],
    ids=[
        "enzymes",
        "ion_channels",
        "gpcr",
        "nuclear_receptors",
    ],
)
def test_all_dataset_loaders(loader_class, tmp_path):
    loader = loader_class(tmp_path)
    X, y = loader.load(as_frame=False)
    assert isinstance(X, list)
    assert isinstance(y, np.ndarray)
    assert isinstance(X[0], np.ndarray)
    assert isinstance(X[1], np.ndarray)
    assert X[0].shape[0] == y.shape[0]
    assert X[1].shape[0] == y.shape[1]