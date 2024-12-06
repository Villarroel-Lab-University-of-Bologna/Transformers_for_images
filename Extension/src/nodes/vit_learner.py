import knime.extension as knext
from utils import knutills as kutil
import logging
from PIL import Image
import pandas as pd
import numpy as np
import base64
from io import BytesIO
from torch.utils.data import Dataset, DataLoader
from transformers import ViTImageProcessor, ViTForImageClassification
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle
from torch.nn.functional import softmax



use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

LOGGER = logging.getLogger(__name__)

# Define custom port objects for KNIME
class OutputPortSpec(knext.PortObjectSpec):
    def __init__(self, spec_data: str) -> None:
        super().__init__()
        self._spec_data = spec_data

    def serialize(self) -> dict:
        return {"spec_data": self._spec_data}

    @classmethod
    def deserialize(cls, data: dict) -> "OutputPortSpec":
        return cls(data["spec_data"])

    @property
    def spec_data(self) -> str:
        return self._spec_data


class OutputPort(knext.PortObject):
    def __init__(self, spec: OutputPortSpec, model) -> None:
        super().__init__(spec)  # Corrected usage of `super()`
        self._model = model

    def serialize(self) -> bytes:
        return pickle.dumps(self._model)

    @classmethod
    def deserialize(cls, spec: OutputPortSpec, data: bytes) -> "MyPortObject":
        return cls(spec, pickle.loads(data))

    @property
    def spec(self) -> OutputPortSpec:
        return OutputPortSpec("vision_transformer_model")


# Define KNIME port types
my_model_port_type = knext.port_type(name="My model port type", object_class=OutputPort, spec_class=OutputPortSpec)

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
    name="Vision Transformer Learner",
    node_type=knext.NodeType.LEARNER,
    icon_path="icons/icon.png",
    category=image_category,
    id="img-model-learner",
)
@knext.input_table(
    name="Train Input Data",
    description="Table containing the training set with image column."
)
@knext.input_table(
    name="Validation Input Data",
    description="Table containing the validation set with image column."
)
@knext.output_port(
    name="Trained Model",
    port_type=my_model_port_type,
    description="Output containing the trained Vision Transformer model."
)

class VisionTransformerLearnerNode:
    """
    Node to train a Vision Transformer model on input image data.
    """

    image_column = knext.ColumnParameter(
        label="Image Column",
        description="Select an Image column for training.",
        port_index=0,
        column_filter=kutil.is_png,
    )

    label_column = knext.ColumnParameter(
        label="Label Column",
        description="Select the column containing class labels.",
        port_index=0,
    )

    num_epochs = knext.IntParameter(
        label="Number of Epochs",
        description="Number of epochs for training the model.",
        default_value=5,
        min_value=1,
    )

    batch_size = knext.IntParameter(
        label="Batch Size",
        description="Batch size for training.",
        default_value=8,
        min_value=1,
    )

    learning_rate = knext.DoubleParameter(
        label="Learning Rate",
        description="Learning rate for the optimizer.",
        default_value=0.001,
        min_value=1e-6,
    )

    class_num = knext.IntParameter(
        label="Number of Classes",
        description="Number of classes to be predicted.",
        default_value=2,
        min_value=2,
    )

    def configure(self, configure_context: knext.ConfigurationContext, input_schema_1: knext.Schema, input_schema_2: knext.Schema):
        image_columns = [(c.name, c.ktype) for c in input_schema_1 if kutil.is_png(c)]
        class_columns = [(c.name, c.ktype) for c in input_schema_1 if kutil.is_numeric(c)]

        if not image_columns:
            raise ValueError("No valid image columns found in the input data.")

        if self.label_column is None:
            self.label_column = class_columns[-1][0]
        if self.image_column is None:
            self.image_column = image_columns[-1][0]

        return OutputPortSpec("vision_transformer_model")

    def execute(self, exec_context: knext.ExecutionContext, input_table_1, input_table_2):
        # Convert input KNIME table to pandas DataFrame
        df_train = input_table_1.to_pandas()
        df_val = input_table_2.to_pandas()

        # Dataset class
        class ImageDataset(Dataset):
            def __init__(self, images, labels, processor=None):
                self.images = images
                self.labels = labels
                self.processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

            def __len__(self):
                return len(self.images)

            def __getitem__(self, idx):
                image = self.images.iloc[idx].convert("RGB")
                label = self.labels.iloc[idx]
                processed_image = self.processor(images=image, return_tensors="pt").pixel_values.squeeze()
                return processed_image.to(device), label

        # Create dataset instances for train and validation splits
        train_dataset = ImageDataset(df_train[self.image_column], df_train[self.label_column])
        val_dataset = ImageDataset(df_val[self.image_column], df_val[self.label_column])

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # Define model, loss, and optimizer
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224').to(device)
        model.config.num_labels = self.class_num
        model.classifier = torch.nn.Linear(in_features=model.config.hidden_size, out_features=self.class_num)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=self.learning_rate)

        # Training loop with validation
        for epoch in range(self.num_epochs):
            # Training Phase
            model.train()
            total_loss_train = 0.0
            total_acc_train = 0

            for train_images, train_labels in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{self.num_epochs}"):
                optimizer.zero_grad()
                outputs = model(pixel_values=train_images).logits
                train_labels = train_labels.to(device)
                loss = criterion(outputs, train_labels.long())
                acc = (outputs.argmax(dim=1) == train_labels).sum().item()

                loss.backward()
                optimizer.step()

                total_loss_train += loss.item()
                total_acc_train += acc

            avg_train_loss = total_loss_train / len(train_loader)
            avg_train_acc = total_acc_train / len(train_dataset)

            # Validation Phase
            model.eval()
            total_loss_val = 0.0
            total_acc_val = 0

            with torch.no_grad():
                for val_images, val_labels in val_loader:
                    outputs = model(pixel_values=val_images).logits
                    val_labels = val_labels.to(device)  # Convert labels to the correct device
                    loss = criterion(outputs, val_labels.long())
                    acc = (outputs.argmax(dim=1) == val_labels).sum().item()

                    total_loss_val += loss.item()
                    total_acc_val += acc

            avg_val_loss = total_loss_val / len(val_loader)
            avg_val_acc = total_acc_val / len(val_dataset)

            LOGGER.info(
                f"Epoch {epoch + 1}/{self.num_epochs} | Train Loss: {avg_train_loss:.3f} | "
                f"Train Acc: {avg_train_acc:.3f} | Val Loss: {avg_val_loss:.3f} | Val Acc: {avg_val_acc:.3f}"
            )

        # Create a MyPortObject with the trained model
        port_object_spec = OutputPortSpec("vision_transformer_model")
        trained_model = OutputPort(port_object_spec, model)

        # Return trained model
        return trained_model
    




    
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
        port_index=1,
        column_filter=kutil.is_png,
    )
    id_column = knext.ColumnParameter(
        label="IDs Column",
        description="Select the ID for images.",
        port_index=1,
        column_filter=kutil.is_string,
    )

    def configure(self, configure_context: knext.ConfigurationContext, input_spec: MyPortObjectSpec, input_schema_2: knext.Schema):
        # Validate the test input table schema (input_schema_2)
        image_columns = [c.name for c in input_schema_2 if kutil.is_png(c)]
        id_columns = [c.name for c in input_schema_2 if kutil.is_string(c)]

        if not image_columns:
            raise ValueError("No valid image columns found in the input data.")
        if not id_columns:
            raise ValueError("No valid ID columns found in the input data.")

        # Set default columns if not selected
        if self.image_column is None:
            self.image_column = image_columns[-1]
        if self.id_column is None:
            self.id_column = id_columns[-1]

        # Placeholder: Assume a maximum of 10 possible classes for the configure step
        placeholder_num_classes = 10
        placeholder_label_columns = [knext.Column(knext.double(), f"Probability_Label_{i}") for i in range(placeholder_num_classes)]

        # Return the expected output schema with placeholder columns
        return input_schema_2.append(
            [
                knext.Column(knext.string(), "ID"),
                knext.Column(knext.string(), "Predicted Class"),
                *placeholder_label_columns,  # Add placeholder columns for probabilities
            ]
        )


    def execute(self, exec_context: knext.ExecutionContext, model_port: MyPortObject, input_table_1):
        # Convert input KNIME table to pandas DataFrame
        df_test = input_table_1.to_pandas()

        # Load the trained model from the input port
        model = model_port._model
        model.eval()
        model.to(device)

        # Determine the actual number of classes dynamically from the model
        num_classes = model.config.num_labels

        # Set up a DataLoader with the test data
        class ImageDataset(Dataset):
            def __init__(self, images, ids):
                self.images = images
                self.ids = ids
                self.processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

            def __len__(self):
                return len(self.images)

            def __getitem__(self, idx):
                image = self.images.iloc[idx].convert("RGB")
                id_ = self.ids.iloc[idx]
                processed_image = self.processor(images=image, return_tensors="pt").pixel_values.squeeze()
                return processed_image.to(device), id_

        # Create the dataset and DataLoader for the test set
        test_dataset = ImageDataset(df_test[self.image_column], df_test[self.id_column])
        test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

        predictions = []
        ids_collected = []
        probabilities_dict = {}

        # Initialize probabilities dictionary with label keys based on the actual number of classes
        for i in range(num_classes):
            probabilities_dict[f"Probability_Label_{i}"] = []

        # Predict with the model
        for images, ids in tqdm(test_dataloader, desc="Predicting"):
            images = images.to(device)
            with torch.no_grad():
                output = model(pixel_values=images).logits
                pred = output.argmax(dim=1).cpu().numpy()
                prob = softmax(output, dim=1).cpu().numpy()

                predictions.extend(pred)
                ids_collected.extend(ids)

                # Split probabilities for each label into individual columns
                for i in range(num_classes):
                    probabilities_dict[f"Probability_Label_{i}"].extend(prob[:, i])

        # Convert predictions to class labels
        class_predictions = [model.config.id2label[pred] for pred in predictions]

        # Creating a DataFrame with predicted classes, IDs, and individual probabilities
        prediction_data = {
            'ID': ids_collected,
            'Predicted Class': class_predictions,
        }
        prediction_data.update(probabilities_dict)

        # Construct a DataFrame, dropping placeholder columns if necessary
        prediction_df = pd.DataFrame(prediction_data)

        # Drop any columns that were added as placeholders but aren't needed
        for i in range(num_classes, 10):  # Assuming a maximum placeholder number of classes was 10
            placeholder_column = f"Probability_Label_{i}"
            if placeholder_column in prediction_df.columns:
                prediction_df.drop(columns=[placeholder_column], inplace=True)

        # Return predictions as KNIME table
        return knext.Table.from_pandas(prediction_df)