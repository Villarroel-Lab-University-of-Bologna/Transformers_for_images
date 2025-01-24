import knime.extension as knext
from utils import knutills as kutil
import logging
from torch.utils.data import Dataset, DataLoader
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    SwinForImageClassification,
    PvtForImageClassification,
    PvtImageProcessor,
    AutoImageProcessor,
)
import torch
from torch.optim import Adam
from sklearn.preprocessing import LabelEncoder


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

LOGGER = logging.getLogger(__name__)


# Define sub-category
image_category = knext.category(
    path="/community/vit_vision",
    level_id="vit",
    name="Transformer model",
    description="Python Nodes for Vision Transformers",
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
    name="Training Data",
    description="Table containing the training set with image column.",
)
@knext.input_table(
    name="Validation Data",
    description="Table containing the validation set with image column.",
)
@knext.output_port(
    name="Trained Model",
    port_type=kutil.classification_model_port_type,
    description="Output containing the trained Vision Transformer model.",
)
class VisionTransformerLearnerNode:
    """

    Learner node


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

    model_choice = knext.StringParameter(
        label="Model Choice",
        description="Choose the Transformer model to use: ViT, Swin Transformer, or Pyramid Transformer.",
        default_value="ViT",
        enum=["ViT", "Swin Transformer", "Pyramid Transformer"],
    )

    def configure(
        self,
        configure_context: knext.ConfigurationContext,
        training_schema: knext.Schema,
        validation_schema: knext.Schema,
    ):


        return self._create_spec(training_schema, validation_schema, class_probability_schema = None)


    def _create_spec(        
        self,
        training_schema: knext.Schema,
        validation_schema: knext.Schema,
        class_probability_schema: knext.Schema,
        ):

        image_column = [
                (c.name, c.ktype) for c in training_schema if kutil.is_png(c)
        ]


       # Check if feature column(s) have been specified
        if not image_column:
            raise knext.InvalidParametersError(
                """Image column has not been specified."""
            )

        
        label_column = [
            (c.name, c.ktype) for c in training_schema if kutil.is_nominal(c)
        ]

        # Check if the target column have been specified
        if len(label_column) == 0:
            raise knext.InvalidParametersError("Target column is missing.")        

        # Create schema from the target column
        target_schema = knext.Schema.from_columns(
            [c for c in training_schema if c.name == label_column[0][0]]
        )

        # Create schema from feature columns
        feature_schema = knext.Schema.from_columns(
            [c for c in training_schema if c.name == image_column[0][0]]
        )

        # Check if the option for predicting class probabilities is enabled
        if class_probability_schema is None:
            class_probability_schema = knext.Schema.from_columns("")

        LOGGER.warning(f"Input Schema: {training_schema}")
        LOGGER.warning(f"Feature Schema: {feature_schema}")
        LOGGER.warning(f"Target Schema: {target_schema}")

        return kutil.ClassificationModelObjectSpec(
            feature_schema,
            target_schema,
            class_probability_schema,
            model_choice=self.model_choice, #TODO make this as path to pre_trained model, either as local path or huggingface.co endpoint
        )

    def execute(
        self, exec_context: knext.ExecutionContext, training_table, validation_table
    ):
        df_train = training_table.to_pandas()
        df_val = validation_table.to_pandas()

        # Extract feature and label columns
        feature_columns = [self.image_column]
        label_column = self.label_column

        LOGGER.warning(feature_columns[0])

        class_probability_schema = None

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
            model = ViTForImageClassification.from_pretrained(
                "google/vit-base-patch16-224"
            )
        elif self.model_choice == "Swin Transformer":
            processor = AutoImageProcessor.from_pretrained(
                "microsoft/swin-base-patch4-window7-224"
            )
            model = SwinForImageClassification.from_pretrained(
                "microsoft/swin-base-patch4-window7-224"
            )
        elif self.model_choice == "Pyramid Transformer":
            processor = PvtImageProcessor.from_pretrained("Zetatech/pvt-medium-224")
            model = PvtForImageClassification.from_pretrained("Zetatech/pvt-medium-224")


        # Get class names(=column names) for probability estimates
        prob_estimates_column_names = kutil.pd.DataFrame(columns=train_labels.unique())

        # Update the table schema with probability estimates column names
        prob_estimates_column_names = prob_estimates_column_names.reindex(
            sorted(prob_estimates_column_names.columns), axis=1
        )

        input_table_schema = training_table.schema
        for column_name in prob_estimates_column_names:
            input_table_schema = input_table_schema.append(
                knext.Column(
                    ktype=knext.double(),
                    name=f"P({self.label_column}_pred={column_name})",
                )
            ).get()


        # Determine the number of classes dynamically
        num_classes = len(label_enc.classes_)

        # Create a schema that contains the class names by getting the last
        # "num_classes" number of columns from the input table.
        # class_probability_schema is then used if user wants to predict
        # probability estimates for the test data.
        if num_classes:
            class_probability_schema = input_table_schema[-num_classes:].get()

        #TODO create schema later
        class ImageDataset(Dataset):
            def __init__(self, image_data, labels, processor):
                self.images = image_data
                self.labels = labels
                self.processor = processor

            def __len__(self):
                return len(self.images)

            def __getitem__(self, idx):
                image = self.images[idx].convert("RGB")
                processed_image = self.processor(
                    images=image, return_tensors="pt"
                ).pixel_values.squeeze(0)
                label = self.labels[idx]
                return processed_image, label

        # Create datasets and data loaders
        train_dataset = ImageDataset(train_images, train_labels_encoded, processor)
        val_dataset = ImageDataset(val_images, val_labels_encoded, processor)

        kutil.check_canceled(exec_context)

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
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

                kutil.check_canceled(exec_context)
                
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
                    val_labels = val_labels.to(
                        device
                    )  # Convert labels to the correct device
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

        # port_object_spec = kutil.ClassificationModelObjectSpec(
        #     feature_schema=knext.Schema.from_columns(
        #         [
        #             knext.Column(name=col, ktype=knext.string())
        #             for col in df_train.columns
        #         ]
        #     ),
        #     target_schema=knext.Schema.from_columns(
        #         [knext.Column(name=self.label_column, ktype=knext.double())]
        #     ),
        #     class_probability_schema=knext.Schema.from_columns(
        #         [
        #             knext.Column(name=f"Probability_Label_{i}", ktype=knext.double())
        #             for i in range(num_classes)
        #         ]
        #     ),
        #     model_choice=self.model_choice,  # Pass model choice
        # )

        trained_model = kutil.ClassificationModelObject(
            spec=self._create_spec(input_table_schema, input_table_schema, class_probability_schema),
            model=model,
            label_enc=label_enc,
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

    prob_columns_suffix = knext.StringParameter(
        "Suffix for probability columns",
        "Allows to add a suffix for the class probability columns.",
        default_value="",
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
    description="Input containing the trained Vision Transformer model.",
)
@knext.input_table(
    name="Test Input Data",
    description="Table containing the test set with image column.",
)
@knext.output_table(
    name="Output table", description="Resulting table with prediction categories."
)
class VisionTransformerPredictor:

    predictor_settings = ClassificationPredictorGeneralSettings()

    def configure(
        self,
        configure_context: knext.ConfigurationContext,
        model_spec: kutil.ClassificationModelObjectSpec,
        table_schema: knext.Schema,
    ):

        y_pred = kutil.get_prediction_column_name(
            self.predictor_settings.prediction_column, model_spec.target_schema
        )

        # Add prediction column names in the schema
        for column_name in y_pred:
            table_schema = table_schema.append(
                knext.Column(ktype=knext.string(), name=column_name)
            )

        # # Define prediction column
        # table_schema = table_schema.append(
        #     knext.Column(ktype=knext.string(), name=prediction_column)
        # )

        # Add probability estimate column names in the schema
        if self.predictor_settings.predict_probs:
            for column in model_spec.class_probability_schema:
                if self.predictor_settings.prob_columns_suffix:
                    table_schema = table_schema.append(
                        knext.Column(
                            ktype=column.ktype,
                            name=f"{column.name}{self.predictor_settings.prob_columns_suffix}",
                        )
                    )
                else:
                    table_schema = table_schema.append(
                        knext.Column(ktype=column.ktype, name=column.name)
                    )

        LOGGER.warning(f"Configured Schema: {[col.name for col in table_schema]}")
        return table_schema

    def execute(
        self,
        exec_context: knext.ExecutionContext,
        model_port: kutil.ClassificationModelObject,
        input_table_1,
    ):
        df_test = input_table_1.to_pandas()

        # Get target column for the prediction column
        y_pred = kutil.get_prediction_column_name(
            self.predictor_settings.prediction_column, model_port.spec.target_schema
        )
        class ImageDataset(Dataset):
            def __init__(self, image_data, processor):
                self.images = image_data
                self.processor = processor

            def __len__(self):
                return len(self.images)

            def __getitem__(self, idx):
                image = self.images.iloc[idx].convert("RGB")
                processed_image = self.processor(
                    images=image, return_tensors="pt"
                ).pixel_values.squeeze(0)
                return processed_image

        LOGGER.warning(model_port.spec.feature_schema)
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
                    probabilities = (
                        torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
                    )
                    all_probabilities.extend(probabilities)


        decoded_predictions = model_port.decode_target_values(
            kutil.pd.Series(all_predictions)
        )

        LOGGER.warning(y_pred)

        LOGGER.warning(df_test)
        decoded_predictions.index = df_test.index
        df_test[y_pred[0]] = decoded_predictions

        if self.predictor_settings.predict_probs:
            class_probability_names = model_port.get_class_probability_column_names(
                y_pred[0]
            )
            for col, probs in zip(class_probability_names, zip(*all_probabilities)):
                df_test[col] = probs

        actual_schema = {col: str(df_test[col].dtype) for col in df_test.columns}
        LOGGER.warning(f"Actual Schema: {actual_schema}")
        return knext.Table.from_pandas(df_test)
