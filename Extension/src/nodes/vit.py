import knime.extension as knext
from utils import knutills as kutil
from utils import modeling_utils as mutil
import logging
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


LOGGER = logging.getLogger(__name__)


# Define sub-category
image_category = knext.category(
    path="/community/vit_ft",
    level_id="vit",
    name="Models",
    description="Vision Transformer fine-tune learner and predictor-",
    icon="icons/icon_Vision Transformer Learner.png",
)


# KNIME node definition for training
@knext.node(
    name="ViT Classification Learner",
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
        column_filter=kutil.is_nominal,
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

    model_choice = knext.EnumParameter(
        label="Model Choice",
        description="Choose the Transformer model to use: ViT, Swin Transformer, or Pyramid Transformer.",
        default_value=mutil.ViTModelSelection.get_default().name,
        enum=mutil.ViTModelSelection,
    )

    def configure(
        self,
        configure_context: knext.ConfigurationContext,
        training_schema: knext.Schema,
        validation_schema: knext.Schema,
    ):

        return self._create_spec(
            training_schema, validation_schema, class_probability_schema=None
        )

    def _create_spec(
        self,
        training_schema: knext.Schema,
        validation_schema: knext.Schema,
        class_probability_schema: knext.Schema,
    ):

        image_column = [(c.name, c.ktype) for c in training_schema if kutil.is_png(c)]

        # Check if image column have been specified
        if not image_column:
            raise knext.InvalidParametersError(
                "PNG image type column in missing in the input table."
            )

        # Preset the left most PNG image column from the input table.
        if self.image_column is None:
            self.image_column = image_column[-1][0]

        # Populate list of target columns
        label_column = [
            (c.name, c.ktype) for c in training_schema if kutil.is_nominal(c)
        ]

        # Check if the target column is available
        if not label_column:
            raise knext.InvalidParametersError("No compatible target column available.")

        # Preset the left most nominal column as target column from the input table.
        if self.label_column is None:
            self.label_column = label_column[-1][0]

        # Create schema from the target column
        target_schema = training_schema[[self.label_column]]

        # Create image column schema from the selected image column
        image_schema = training_schema[[self.image_column]]

        # Check if the option for predicting class probabilities is enabled
        if class_probability_schema is None:
            class_probability_schema = knext.Schema.from_columns("")

        # TODO: add a check that the validation schema must contain the same columns as selected for training

        return kutil.ViTClassificationModelObjectSpec(
            image_schema,
            target_schema,
            class_probability_schema,
            model_choice=self.model_choice,  # TODO make this as path to pre_trained model, either as local path or huggingface.co endpoint
        )

    def execute(
        self, exec_context: knext.ExecutionContext, training_table, validation_table
    ):
        df_train = training_table.to_pandas()
        df_val = validation_table.to_pandas()

        # Extract feature and label columns
        image_column = self.image_column
        label_column = self.label_column

        class_probability_schema = None

        # Extract images and labels from both training and validation input table.
        train_images = df_train[image_column]
        train_labels = df_train[label_column]

        val_images = df_val[image_column]
        val_labels = df_val[label_column]

        # Initialize a label encoder and encode labels
        label_enc = kutil.LabelEncoder()
        train_labels_encoded = label_enc.fit_transform(train_labels)
        val_labels_encoded = label_enc.transform(val_labels)
        
        LOGGER.warn(self.model_choice)


        # Model selection logic
        if self.model_choice == mutil.ViTModelSelection.ViT.name:
            processor = ViTImageProcessor.from_pretrained(
                mutil.ViTModelSelection.ViT.description
            )
            model = ViTForImageClassification.from_pretrained(
                mutil.ViTModelSelection.ViT.description
            )
        elif self.model_choice == mutil.ViTModelSelection.SWIN.name:
            processor = AutoImageProcessor.from_pretrained(
                mutil.ViTModelSelection.SWIN.description
            )
            model = SwinForImageClassification.from_pretrained(
                mutil.ViTModelSelection.SWIN.description
            )
        elif self.model_choice == mutil.ViTModelSelection.PYRAMID.name:
            processor = PvtImageProcessor.from_pretrained(
                mutil.ViTModelSelection.PYRAMID.description
            )
            model = PvtForImageClassification.from_pretrained(
                mutil.ViTModelSelection.PYRAMID.description
            )

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

        # Create datasets and data loaders
        train_dataset = mutil.TrainingImageDataset(
            train_images, train_labels_encoded, processor
        )
        val_dataset = mutil.TrainingImageDataset(
            val_images, val_labels_encoded, processor
        )

        kutil.check_canceled(exec_context)

        train_loader = mutil.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_loader = mutil.DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False
        )

        exec_context.set_progress(0.3)

        # Model setup
        model.config.num_labels = num_classes
        if self.model_choice == mutil.ViTModelSelection.PYRAMID.name:
            model.classifier = torch.nn.Linear(model.config.hidden_sizes[-1], num_classes)
        else:
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
                train_labels = train_labels.to(mutil.device)
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
                        mutil.device
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

        trained_model = kutil.ViTClassificationModelObject(
            spec=self._create_spec(
                input_table_schema, input_table_schema, class_probability_schema
            ),
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
    name="ViT Classification Predictor",
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
        model_spec: kutil.ViTClassificationModelObjectSpec,
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

        return table_schema

    def execute(
        self,
        exec_context: knext.ExecutionContext,
        model_port: kutil.ViTClassificationModelObject,
        input_table_1,
    ):
        df_test = input_table_1.to_pandas()

        # Get list target column for the prediction column
        y_pred = kutil.get_prediction_column_name(
            self.predictor_settings.prediction_column, model_port.spec.target_schema
        )

        image_col = model_port.spec.image_schema.column_names

        features = df_test[image_col]

        LOGGER.warning(image_col[0])

        images = features[image_col[0]]


        if self.model_choice == mutil.ViTModelSelection.ViT.name:
            processor = ViTImageProcessor.from_pretrained(
                mutil.ViTModelSelection.ViT.description
            )
        elif self.model_choice == mutil.ViTModelSelection.SWIN.name:
            processor = AutoImageProcessor.from_pretrained(
                mutil.ViTModelSelection.SWIN.description
            )
        elif self.model_choice == mutil.ViTModelSelection.PYRAMID.name:
            processor = PvtImageProcessor.from_pretrained(
                mutil.ViTModelSelection.PYRAMID.description
            )

        dataset = mutil.PredictionImageDataset(images, processor)
        dataloader = mutil.DataLoader(dataset, batch_size=8, shuffle=False)

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

        decoded_predictions.index = df_test.index
        df_test[y_pred[0]] = decoded_predictions

        if self.predictor_settings.predict_probs:

            if self.predictor_settings.prediction_column:
                # Get original target column name (and not the custom name)
                # for class probability column names
                y_pred = kutil.get_prediction_column_name(
                    "", model_port.spec.target_schema
                )

            class_probability_names = model_port.get_class_probability_column_names(
                y_pred, self.predictor_settings.prob_columns_suffix
            )
            for col, probs in zip(class_probability_names, zip(*all_probabilities)):
                df_test[col] = probs

        return knext.Table.from_pandas(df_test)
