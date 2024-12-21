import pydantic


class DigitFeature(pydantic.BaseModel):
    quotes: list[str] | None = pydantic.Field(
        default=None,
        description="The exact sentences that contain the value. This property is None if the feature was not found in the description.",
    )
    value: int | None = pydantic.Field(
        default=None,
        description="The exact integer value in the sentences. This property is None if the feature was not found in the description",
    )


class StringFeature(pydantic.BaseModel):
    quotes: list[str] | None = pydantic.Field(
        default=None,
        description="The exact sentences that contain the value. This property is None if the feature was not found in the description.",
    )
    value: str | None = pydantic.Field(
        description="The exact value in the sentences. This property is None if the feature was not found in the description"
    )


class BooleanFeature(pydantic.BaseModel):
    quotes: list[str] | None = pydantic.Field(
        default=None,
        description="The exact sentences that contain the value. This property is None if the feature was not found in the description.",
    )
    value: bool | None = pydantic.Field(
        default=None,
        description="Whether the house has this feature. This property is None if the feature was not found in the description",
    )
