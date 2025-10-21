
from pydantic import BaseModel, ConfigDict, Field, field_validator, ValidationError
from typing import Optional


class DataExample(BaseModel):
    age: float = Field(..., ge=-1)


class MLData(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra='allow',
        str_to_lower=True,
        str_strip_whitespace=True
    )

    age: int = Field(-1, ge=0, le=120)
    name: str = Field("UNKNOWN")
    income: Optional[float] = Field(None, ge=-1, le=10_000_000)
    credit_score: Optional[float] = Field(None, ge=300, le=850)

    @field_validator('age')
    @classmethod
    def validate_age(cls, v: int) -> int:
        """Custom age validation beyond basic constraints."""
        if v == 0:
            print("age = 0")
        elif v > 100:
            print("Age > 100")
        return v

try:
    usr = MLData(age="78",
                 name="JANE Austin ",
                 gender="F")
    print(usr.model_dump())
    print(usr.model_json_schema())

except ValidationError as e:
    print(e)


