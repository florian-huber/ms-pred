import pytest
import requests
import zipfile
from ms_pred.dag_pred import joint_model
import pandas as pd

@pytest.fixture(scope="session")
def download_model(tmp_path_factory):
    url = "https://zenodo.org/record/8433354/files/canopus_iceberg_models.zip"
    tmp_path = tmp_path_factory.mktemp("data")  # create a temporary directory
    zip_path = tmp_path / "canopus_iceberg_models.zip"

    # Download the file
    response = requests.get(url)
    with open(zip_path, 'wb') as file:
        file.write(response.content)

    # Unzip the file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(tmp_path)

    return tmp_path

@pytest.fixture(scope="module")
def load_model(download_model):
    model_path = download_model
    inten_ckpt = model_path / "canopus_iceberg_score.ckpt"
    gen_ckpt = model_path / "canopus_iceberg_generate.ckpt"

    model = joint_model.JointModel.from_checkpoints(
        inten_checkpoint=str(inten_ckpt), gen_checkpoint=str(gen_ckpt)
    )
    return model

#def test_sample_data():
#    test_df = pd.read_csv("../data/spec_datasets/sample_labels.tsv", sep="\t")
#    assert test_df.shape == (20, 3)

def test_model_prediction(load_model):
    #test_df = pd.read_csv("../data/spec_datasets/sample_labels.tsv", sep="\t")

    model = load_model

    outputs = model.predict_mol(
        smi="CCOC(=O)c1ccc(NC(=S)NC(=O)c2ccccc2)cc1",#test_df.iloc[2]["smiles"],
        adduct="[M+Na]+", #test_df.iloc[2]["ionization"],
        device="cpu",
        max_nodes=100,
        binned_out=False,
        threshold=0,
    )

    hash1 = '9ae5bad23bcb385d7c01c6ced2768d0d2b90b7296b11f839c701905ee9058382'
    assert outputs["frags"][hash1]["atoms_pulled"][:5] == [17, 18, 21, 1, 16]
