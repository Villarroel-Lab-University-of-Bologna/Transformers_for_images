import knime.extension as knext
from utils import knutills as kutil
import logging
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import ViTImageProcessor
import torch
from tqdm import tqdm
import pandas as pd
from torch.nn.functional import softmax
import pickle

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

LOGGER = logging.getLogger(__name__)

# Define custom port objects for KNIME
class MyPortObjectSpec(knext.PortObjectSpec):
    def __init__(self, spec_data: str) -> None:
        super().__init__()
        self._spec_data = spec_data

    def serialize(self) -> dict:
        return {"spec_data": self._spec_data}

    @classmethod
    def deserialize(cls, data: dict) -> "MyPortObjectSpec":
        return cls(data["spec_data"])

    @property
    def spec_data(self) -> str:
        return self._spec_data


class MyPortObject(knext.PortObject):
    def __init__(self, spec: MyPortObjectSpec, model) -> None:
        super().__init__(spec)  # Corrected usage of `super()`
        self._model = model

    def serialize(self) -> bytes:
        return pickle.dumps(self._model)

    @classmethod
    def deserialize(cls, spec: MyPortObjectSpec, data: bytes) -> "MyPortObject":
        return cls(spec, pickle.loads(data))

    @property
    def spec(self) -> MyPortObjectSpec:
        return self._spec

# Define KNIME port types - this should be defined in a single place and used consistently
my_model_port_type = knext.port_type(name="My model port type", object_class=MyPortObject, spec_class=MyPortObjectSpec)

# Define sub-category
image_category = knext.category(
    path="/community/demo",
    level_id="demoimg",
    name="Images",
    description="Python Nodes for Vision Transformer",
    icon="icons/icon.png",
)

# KNIME node definition
@knext.node(
    name="Vision Transformer Predictor",
    node_type=knext.NodeType.PREDICTOR,
    icon_path="icons/icon.png",
    category=image_category,
    id="img-model-predictor",
)
@knext.input_port(
    name="Trained Model",
    port_type=my_model_port_type,
    description="Input containing the trained Vision Transformer model."
)
@knext.input_table(
    name="Test Input Data",
    description="Table containing the test set with image column."
)
@knext.output_table(
    name = "Output table",
    description =  "Resulting table with prediction categories"
)

class VisionTransformerPredictor:

    image_column = knext.ColumnParameter(
        label="Image Column",
        description="Select an Image column for prediction.",
        port_index=0,
        column_filter=kutil.is_png,
    )
    id_column = knext.ColumnParameter(
        label="IDs Column",
        description="Select the ID for images.",
        port_index=0,
        column_filter=kutil.is_string,
    )

    def configure(self, configure_context: knext.ConfigurationContext, input_schema_1: knext.Schema):
        image_columns = [(c.name, c.ktype) for c in input_schema_1 if kutil.is_png(c)]
        id_columns = [(c.name, c.ktype) for c in input_schema_1 if kutil.is_string(c)]

        if not image_columns:
            raise ValueError("No valid image columns found in the input data.")

        if self.image_column is None:
            self.image_column = image_columns[-1][0]
        if self.id_column is None:
            self.id_column = id_columns[-1][0]

        
        return input_schema_1.append(
            [
                knext.Column(knext.list_(knext.double()), "Probabilities"),
                knext.Column(knext.list_(knext.int64()), "Predicted Class"),
            ]
        )

    def execute(self, exec_context: knext.ExecutionContext, input_table_1, model_port: MyPortObject):
        # Convert input KNIME table to pandas DataFrame
        df_test = input_table_1.to_pandas()

        # Load the trained model from the input port
        model = model_port._model
        model.eval()
        model.to(device)

        # Get batch size from the model if available
        batch_size = getattr(model, 'batch_size', 8)  # Default to 8 if not available

        # Dataset class
        class ImageDataset(Dataset):
            def __init__(self, images, id, processor=None):
                self.images = images
                self.id = id
                self.processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

            def __len__(self):
                return len(self.images)

            def __getitem__(self, idx):
                image = self.images.iloc[idx].convert("RGB")
                id = self.id.iloc[idx]
                processed_image = self.processor(images=image, return_tensors="pt").pixel_values.squeeze()
                return processed_image.to(device), id
            
        test_dataset = ImageDataset(df_test[self.image_column], df_test[self.id_column])
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        predictions = []
        ids_collected = []  # List to collect post IDs
        probabilities = []  # List to collect probabilities

        # Prediction loop
        for images, ids in tqdm(test_dataloader, desc="Predicting"):
            images = images.to(device)

            with torch.no_grad():
                output = model(pixel_values=images).logits
                pred = output.argmax(dim=1).detach().cpu().numpy()
                prob = softmax(output, dim=1).cpu().numpy()
                predictions.extend(pred)
                probabilities.extend([list(map(lambda x: round(x, 2), p)) for p in prob])

        class_predictions = [model.config.id2label[pred] for pred in predictions]

        # Creating a DataFrame with predicted classes and probabilities
        prediction_df = pd.DataFrame({
            'Predicted class': class_predictions,
            'Probabilities': probabilities
        })

        # Return predictions as KNIME table
        return knext.Table.from_pandas(prediction_df)