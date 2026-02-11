"""Converters - Auto-generate Spark schemas from Pydantic models

This module ensures Single Source of Truth by automatically converting
Pydantic models to Spark StructType schemas.
"""

from typing import Type, get_args, get_origin
from pydantic import BaseModel
from datetime import datetime, date
from enum import Enum


def pydantic_to_spark_schema(model: Type[BaseModel]) -> "StructType":
    """
    Automatically convert a Pydantic Model to Spark StructType

    This is the key converter that ensures Schema Single Source of Truth.

    Args:
        model: Pydantic model class

    Returns:
        Spark StructType schema

    Example:
        >>> from shrekommender_common.schemas.events import UnifiedEventRecord
        >>> schema = pydantic_to_spark_schema(UnifiedEventRecord)
    """
    from pyspark.sql.types import (
        StructType, StructField,
        StringType, IntegerType, LongType, FloatType, DoubleType,
        BooleanType, TimestampType, DateType, ArrayType
    )

    # Type mapping from Python to Spark
    TYPE_MAPPING = {
        str: StringType(),
        int: IntegerType(),
        float: DoubleType(),
        bool: BooleanType(),
        datetime: TimestampType(),
        date: DateType(),
    }

    fields = []

    for field_name, field_info in model.model_fields.items():
        field_type = field_info.annotation
        nullable = True

        # Handle Optional[T] and Union types
        origin = get_origin(field_type)
        if origin is not None:
            args = get_args(field_type)
            # Optional[X] is equivalent to Union[X, None]
            if type(None) in args:
                nullable = True
                # Get the non-None type
                field_type = next((t for t in args if t is not type(None)), str)
            else:
                # For Literal types, get the underlying type
                if len(args) > 0:
                    # For Literal[EventType.WATCH], get the type of the value
                    field_type = type(args[0]) if not isinstance(args[0], type) else args[0]

        # Handle Required fields (no default value and not Optional)
        if field_info.is_required() and not nullable:
            nullable = False

        # Map Python type to Spark type
        if field_type in TYPE_MAPPING:
            spark_type = TYPE_MAPPING[field_type]
        elif isinstance(field_type, type) and issubclass(field_type, Enum):
            # Enum types map to String
            spark_type = StringType()
        else:
            # Default to String for unknown types
            spark_type = StringType()

        # Get field description for metadata
        description = field_info.description or ""

        fields.append(
            StructField(
                field_name,
                spark_type,
                nullable=nullable,
                metadata={"description": description}
            )
        )

    return StructType(fields)


def get_unified_event_spark_schema() -> "StructType":
    """
    Get the Spark schema for the unified event table

    This schema is auto-generated from UnifiedEventRecord Pydantic model.

    Returns:
        Spark StructType for unified event records
    """
    from .events import UnifiedEventRecord
    return pydantic_to_spark_schema(UnifiedEventRecord)
