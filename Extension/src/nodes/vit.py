import knime.extension as knext
from utils import knutills as kutil
import logging
from PIL import Image
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import ViTImageProcessor, ViTForImageClassification, SwinForImageClassification, PvtForImageClassification, PvtImageProcessor, AutoImageProcessor
import torch
from torch.optim import Adam
import pickle
from torch.nn.functional import softmax
from sklearn.preprocessing import LabelEncoder


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

LOGGER = logging.getLogger(__name__)


# Define sub-category
image_category = knext.category(
    path="/community/demo",
    level_id="demoimg",
    name="Transformer model",
    description="Python Nodes for Vision Transformer",
    icon="icons/icon_Vision Transformer Learner.png",
)

# KNIME node definition for training
@knext.node(
    name="Transformer Learner",
    node_type=knext.NodeType.LEARNER,
    icon_path="icons/icon_Vision Transformer Learner.png",
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
    port_type=kutil.classification_model_port_type,
    description="Output containing the trained Vision Transformer model."
)



class VisionTransformerLearnerNode:
    '''

    Learner node


    '''
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

    model_choice = knext.StringParameter(
        label="Model Choice",
        description="Choose the Transformer model to use: ViT, Swin Transformer, or Pyramid Transformer.",
        default_value="ViT",
        enum = ["ViT", "Swin Transformer", "Pyramid Transformer"]
    )


    def configure(self, configure_context: knext.ConfigurationContext, input_schema_1: knext.Schema, class_probability_schema: knext.Schema):
        image_column = [(c.name, c.ktype) for c in input_schema_1 if kutil.is_png(c)]
        label_column = [(c.name, c.ktype) for c in input_schema_1 if kutil.is_numeric(c)]

        if not image_column:
            raise ValueError("No valid image columns found in the input data.")

        # Create schema from the target column
        target_schema = knext.Schema.from_columns(
            [c for c in input_schema_1 if c.name == label_column[0][0]]
        )


        # Create schema from feature columns
        feature_schema = knext.Schema.from_columns(
            [c for c in input_schema_1 if c.name in image_column]
        )

        # Check if feature column(s) have been specified
        if not image_column:
            raise knext.InvalidParametersError(
                """Feature column(s) have not been specified."""
            )

        # Check if the target column have been specified
        if not label_column:
            raise knext.InvalidParametersError(
                """Target column has not been specified."""
            )

        # Check if the option for predicting class probabilities is enabled
        if class_probability_schema is None:
            class_probability_schema = knext.Schema.from_columns("")


        LOGGER.info(f"Input Schema: {input_schema_1}")
        LOGGER.info(f"Feature Schema: {feature_schema}")
        LOGGER.info(f"Target Schema: {target_schema}")

        return kutil.ClassificationModelObjectSpec(feature_schema, target_schema, class_probability_schema, model_choice=self.model_choice)

    def execute(self, exec_context: knext.ExecutionContext, input_table_1, input_table_2):
        df_train = input_table_1.to_pandas()
        df_val = input_table_2.to_pandas()

        # Extract feature and label columns
        feature_columns = [self.image_column]
        label_column = self.label_column

        # Extract images and labels
        train_images = df_train[feature_columns[0]]
        train_labels = df_train[label_column]
        val_images = df_val[feature_columns[0]]
        val_labels = df_val[label_column]

        # Initialize a label encoder and encode labels
        label_enc = LabelEncoder()
        train_labels_encoded = label_enc.fit_transform(train_labels)
        val_labels_encoded = label_enc.transform(val_labels)

        # Model selection logic
        
        if self.model_choice == "ViT":
            processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
            model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
        elif self.model_choice == "Swin Transformer":
            processor = AutoImageProcessor.from_pretrained('microsoft/swin-base-patch4-window7-224')
            model = SwinForImageClassification.from_pretrained('microsoft/swin-base-patch4-window7-224')
        elif self.model_choice == "Pyramid Transformer":
            processor = PvtImageProcessor.from_pretrained('Zetatech/pvt-medium-224')
            model = PvtForImageClassification.from_pretrained('Zetatech/pvt-medium-224')
        

        class ImageDataset(Dataset):
            def __init__(self, image_data, labels, processor):
                self.images = image_data
                self.labels = labels
                self.processor = processor

            def __len__(self):
                return len(self.images)

            def __getitem__(self, idx):
                image = self.images[idx].convert("RGB")
                processed_image = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
                label = self.labels[idx]
                return processed_image, label

        # Create datasets and data loaders
        train_dataset = ImageDataset(train_images, train_labels_encoded, processor)
        val_dataset = ImageDataset(val_images, val_labels_encoded, processor)
        
        kutil.check_canceled(exec_context)

        # Determine the number of classes dynamically
        num_classes = len(label_enc.classes_)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        exec_context.set_progress(0.3)

        # Model setup
        model.config.num_labels = num_classes
        model.classifier = torch.nn.Linear(model.config.hidden_size, num_classes)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=self.learning_rate)

        kutil.check_canceled(exec_context)
        # Training loop with validation
        for epoch in range(self.num_epochs):
            # Training Phase
            model.train()
            total_loss_train = 0.0
            total_acc_train = 0

            kutil.check_canceled(exec_context)

            for train_images, train_labels in train_loader:
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

            exec_context.set_progress(0.6)

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

            exec_context.set_progress(0.8)

            avg_val_loss = total_loss_val / len(val_loader)
            avg_val_acc = total_acc_val / len(val_dataset)

            LOGGER.info(
                f"Epoch {epoch + 1}/{self.num_epochs} | Train Loss: {avg_train_loss:.3f} | "
                f"Train Acc: {avg_train_acc:.3f} | Val Loss: {avg_val_loss:.3f} | Val Acc: {avg_val_acc:.3f}"
            )

        port_object_spec = kutil.ClassificationModelObjectSpec(
            feature_schema=knext.Schema.from_columns([knext.Column(name=col, ktype=knext.string()) for col in df_train.columns]),
            target_schema=knext.Schema.from_columns([knext.Column(name=self.label_column, ktype=knext.double())]),
            class_probability_schema=knext.Schema.from_columns([knext.Column(name=f"Probability_Label_{i}", ktype=knext.double()) for i in range(num_classes)]),
            model_choice=self.model_choice,  # Pass model choice
        )



        trained_model = kutil.ClassificationModelObject(
            spec=port_object_spec,
            model=model,
            label_enc=label_enc,
            one_hot_encoder=None,
            missing_value_handling_setting=None
        )

        return trained_model
    



# General settings for classification predictor node
@knext.parameter_group(label="Output")
class ClassificationPredictorGeneralSettings:

    prediction_column = knext.StringParameter(
        "Custom prediction column name",
        "If no name is specified for the prediction column, it will default to `<target_column_name>_pred`.",
        default_value="",
    )

    predict_probs = knext.BoolParameter(
        "Predict probability estimates",
        "Predict probability estimates for each target class.",
        True,
    )

# KNIME node definition for prediction
@knext.node(
    name="Transformer Predictor",
    node_type=knext.NodeType.PREDICTOR,
    icon_path="icons/icon_Vision Transformer Predictor.png",
    category=image_category,
    id="img-model-predictor",
)
@knext.input_port(
    name="Trained Model",
    port_type=kutil.classification_model_port_type,
    description="Input containing the trained Vision Transformer model."
)
@knext.input_table(
    name="Test Input Data",
    description="Table containing the test set with image column."
)
@knext.output_table(
    name="Output table",
    description="Resulting table with prediction categories."
)

class VisionTransformerPredictor:

    predictor_settings = ClassificationPredictorGeneralSettings()

    def configure(self, configure_context: knext.ConfigurationContext, model_spec: kutil.ClassificationModelObjectSpec, table_schema: knext.Schema):
        prediction_column = self.predictor_settings.prediction_column or "Prediction Category"

        # Define prediction column
        table_schema = table_schema.append(
            knext.Column(ktype=knext.string(), name=prediction_column)
        )

        # Add probability columns if requested
        if self.predictor_settings.predict_probs:
            class_labels = [col.name for col in model_spec.target_schema]
            LOGGER.warning(f"Class Labels for Probabilities: {class_labels}")
            
            for class_label in class_labels:
                column_name = f"Probability_Label_{class_label}"
                table_schema = table_schema.append(
                    knext.Column(ktype=knext.double(), name=column_name)
                )

        LOGGER.warning(f"Configured Schema: {[col.name for col in table_schema]}")
        return table_schema

    def execute(self, exec_context: knext.ExecutionContext, model_port: kutil.ClassificationModelObject, input_table_1):
        df_test = input_table_1.to_pandas()

        class ImageDataset(Dataset):
            def __init__(self, image_data, processor):
                self.images = image_data
                self.processor = processor

            def __len__(self):
                return len(self.images)

            def __getitem__(self, idx):
                image = self.images.iloc[idx].convert("RGB")
                processed_image = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
                return processed_image

        feature_columns = model_port.spec.feature_schema.column_names
        features = df_test[feature_columns]
        images = features[feature_columns[0]]

        processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

        dataset = ImageDataset(images, processor)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

        model_port._model.eval()
        all_predictions = []
        all_probabilities = []

        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(model_port._model.device)
                outputs = model_port._model(pixel_values=batch).logits
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()
                all_predictions.extend(predictions)

                if self.predictor_settings.predict_probs:
                    probabilities = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
                    all_probabilities.extend(probabilities)

        prediction_column_name = self.predictor_settings.prediction_column or "Prediction Category"
        decoded_predictions = model_port.decode_target_values(pd.Series(all_predictions))
        decoded_predictions.index = df_test.index
        df_test[prediction_column_name] = decoded_predictions

        if self.predictor_settings.predict_probs:
            class_probability_names = model_port.get_class_probability_column_names([prediction_column_name])
            for col, probs in zip(class_probability_names, zip(*all_probabilities)):
                df_test[col] = probs

        actual_schema = {col: str(df_test[col].dtype) for col in df_test.columns}
        LOGGER.warning(f"Actual Schema: {actual_schema}")
        return knext.Table.from_pandas(df_test)