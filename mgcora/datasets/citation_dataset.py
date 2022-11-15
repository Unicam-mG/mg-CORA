from spektral.datasets import Citation
from spektral.layers import GCNConv
from spektral.transforms import LayerPreprocess


def get_dataset(dataset_enum):
    return Citation(dataset_enum.value, normalize_x=True, transforms=[LayerPreprocess(GCNConv)])
