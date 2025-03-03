import knime.extension as knext
import logging
import pickle
from sklearn.preprocessing import LabelEncoder
import pandas as pd


LOGGER = logging.getLogger(__name__)

# pre-define values factory strings
# link to all supported ValueFactoryStrings
# https://github.com/knime/knime-python/blob/49aeaba3819edd635519641c81cc7f9541cf090e/org.knime.python3.arrow.types/plugin.xml

ZONED_DATE_TIME_ZONE_VALUE = "org.knime.core.data.v2.time.ZonedDateTimeValueFactory2"
LOCAL_TIME_VALUE = "org.knime.core.data.v2.time.LocalTimeValueFactory"
LOCAL_DATE_VALUE = "org.knime.core.data.v2.time.LocalDateValueFactory"
LOCAL_DATE_TIME_VALUE = "org.knime.core.data.v2.time.LocalDateTimeValueFactory"

PNG_IMAGE_VALUE = "org.knime.core.data.image.png.PNGImageValueFactory"


def is_zoned_datetime(column: knext.Column) -> bool:
    """
    Checks if date&time column has the timezone or not.
    @return: True if selected date&time column has time zone
    """
    return __is_type_x(column, ZONED_DATE_TIME_ZONE_VALUE)


def is_datetime(column: knext.Column) -> bool:
    """
    Checks if a column is of type Date&Time.
    @return: True if selected column is of type date&time
    """
    return __is_type_x(column, LOCAL_DATE_TIME_VALUE)


def is_time(column: knext.Column) -> bool:
    """
    Checks if a column is of type Time only.
    @return: True if selected column has only time.
    """
    return __is_type_x(column, LOCAL_TIME_VALUE)


def is_date(column: knext.Column) -> bool:
    """
    Checks if a column is of type date only.
    @return: True if selected column has date only.
    """
    return __is_type_x(column, LOCAL_DATE_VALUE)


def boolean_or(*functions):
    """
    Return True if any of the given functions returns True
    @return: True if any of the functions returns True
    """

    def new_function(*args, **kwargs):
        return any(f(*args, **kwargs) for f in functions)

    return new_function


def is_type_timestamp(column: knext.Column):
    """
    This function checks on all the supported timestamp columns in KNIME.
    Note that legacy date&time types are not supported.
    @return: True if timestamp column is compatible with the respective logical types supported in KNIME.
    """

    return boolean_or(is_time, is_date, is_datetime, is_zoned_datetime)(column)


def __is_type_x(column: knext.Column, type: str) -> bool:
    """
    Checks if column contains the given type
    @return: True if Column Type is of that type
    """
    return (
        isinstance(column.ktype, knext.LogicalType)
        and type in column.ktype.logical_type
    )


def is_string(column: knext.Column) -> bool:
    """
    Check if column is of type string
    @return: True if Column Type is of type string
    """
    return column.ktype == knext.string()


def is_nominal(column: knext.Column) -> bool:
    """
    Check if column is a nominal datatype (string or a boolean)
    @return: True if Column Type is of type string or a boolean
    """
    return column.ktype == knext.string() or column.ktype == knext.bool_()


def is_numeric(column: knext.Column) -> bool:
    """
    Checks if column is numeric e.g. int, long or double.
    @return: True if Column is numeric
    """
    return (
        column.ktype == knext.double()
        or column.ktype == knext.int32()
        or column.ktype == knext.int64()
    )


def is_boolean(column: knext.Column) -> bool:
    """
    Checks if column is boolean
    @return: True if Column is boolean
    """
    return column.ktype == knext.boolean()


def is_numeric_or_string(column: knext.Column) -> bool:
    """
    Checks if column is numeric or string
    @return: True if Column is numeric or string
    """
    return boolean_or(is_numeric, is_string)(column)


def is_int_or_string(column: knext.Column) -> bool:
    """
    Checks if column is int or string
    @return: True if Column is numeric or string
    """
    return column.ktype in [
        knext.int32(),
        knext.int64(),
        knext.string(),
    ]


def is_binary(column: knext.Column) -> bool:
    """
    Checks if column is of binary object
    @return: True if Column is binary object
    """
    return column.ktype == knext.blob()


def is_png(column: knext.Column) -> bool:
    """
    Checks if column contains PNG image
    @return: True if Column is image
    """
    return __is_type_x(column, PNG_IMAGE_VALUE)


def check_canceled(exec_context: knext.ExecutionContext) -> None:
    """
    Checks if the user has canceled the execution and if so throws a RuntimeException
    """
    if exec_context.is_canceled():
        raise RuntimeError("Execution canceled")


def get_prediction_column_name(prediction_column, target_schema):
    """
    Adds "_pred" suffix to prediction column names.
    @returns: A list of all the columns in the target schema
    """
    if prediction_column.strip() != "":
        y_pred = [prediction_column for y in target_schema.column_names]
    else:
        y_pred = [f"{y}_pred" for y in target_schema.column_names]


    return y_pred


def concatenate_predictions_with_input_table(df, dfx_predictions):
    """
    Append the prediction column with input data frame
    @returns: Updated dataframe with concatenated prediction column.
    """
    # Replace index with original dataframe
    dfx_predictions.index = df.index

    # Horizontally append a column
    df = pd.concat([df, dfx_predictions], axis=1)

    return df


class ViTClassificationModelObjectSpec(knext.PortObjectSpec):
    """
    Specification of ViT model port for classification.
    """

    def __init__(
        self,
        image_schema: knext.Schema,
        target_schema: knext.Schema,
        class_probability_schema: knext.Schema,
        model_choice: str,
    ) -> None:
        self._image_schema = image_schema
        self._target_schema = target_schema
        self._class_probability_schema = class_probability_schema
        self._model_choice = model_choice

    def serialize(self) -> dict:
        return {
            "image_schema": self._image_schema.serialize(),
            "target_schema": self._target_schema.serialize(),
            "class_probability_schema": self._class_probability_schema.serialize(),
            "model_choice": self._model_choice,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "ViTClassificationModelObjectSpec":
        return cls(
            knext.Schema.deserialize(data["image_schema"]),
            knext.Schema.deserialize(data["target_schema"]),
            knext.Schema.deserialize(data["class_probability_schema"]),
            data["model_choice"],
        )

    @property
    def image_schema(self) -> knext.Schema:
        return self._image_schema

    @property
    def target_schema(self) -> knext.Schema:
        return self._target_schema

    @property
    def class_probability_schema(self) -> knext.Schema:
        return self._class_probability_schema

    @property
    def model_choice(self) -> str:
        return self._model_choice


class ViTClassificationModelObject(knext.PortObject):
    def __init__(
        self,
        spec: ViTClassificationModelObjectSpec,
        model,
        label_enc,
    ) -> None:
        super().__init__(spec)
        self._model = model
        self._label_enc = label_enc

    def serialize(self) -> bytes:
        return pickle.dumps(
            (
                self._model,
                self._label_enc,
            )
        )

    @property
    def spec(self) -> ViTClassificationModelObjectSpec:
        return super().spec

    @property
    def one_hot_encoder(self) -> LabelEncoder:
        return self._label_enc

    @classmethod
    def deserialize(
        cls, spec: ViTClassificationModelObjectSpec, data: bytes
    ) -> "ViTClassificationModelObject":
        (
            model,
            label_encoder,
        ) = pickle.loads(data)
        return cls(spec, model, label_encoder)

    def decode_target_values(self, predicted_column):

        if self._label_enc is None:
            raise ValueError(
                "Label encoder is not set. Ensure the encoder is passed during training."
            )

        le_name_mapping = dict(
            zip(range(len(self._label_enc.classes_)), self._label_enc.classes_)
        )

        decoded_column = predicted_column.replace(le_name_mapping)
        return decoded_column

    def get_class_probability_column_names(self, predicted_column_name, suffix):

        # Class probability column names are adjusted to “P({predicted_col_name}={class_name})”
        class_probability_column_names = self._label_enc.classes_
        for i in range(len(class_probability_column_names)):
            class_probability_column_names[i] = (
                f"P({predicted_column_name[0]}={class_probability_column_names[i]}){suffix}"
            )

        return class_probability_column_names



classification_model_port_type = knext.port_type(
    name="ViT Classification Predictor model port type",
    object_class=ViTClassificationModelObject,
    spec_class=ViTClassificationModelObjectSpec,
)
