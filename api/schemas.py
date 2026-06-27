"""
Pydantic schemas for FastAPI request validation and response serialization.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import date


class PredictionRequest(BaseModel):
    """Input payload for a single return prediction."""

    Customer_ID: str = Field(default="CUST_00001", example="CUST_00001")
    ZIP_Code: str = Field(default="10001", example="10001")
    Account_Age_Days: int = Field(default=365, ge=1, example=365)
    Past_Purchases: int = Field(default=10, ge=1, example=10)
    Past_Returns: int = Field(default=1, ge=0, example=1)
    Product_ID: str = Field(default="PROD_1234", example="PROD_1234")
    Category: str = Field(default="Clothing", example="Clothing")
    Price: float = Field(default=79.99, gt=0, example=79.99)
    Discount_Percentage: float = Field(default=20.0, ge=0, le=100, example=20.0)
    Shipping_Method: str = Field(default="Standard", example="Standard")
    Carrier: str = Field(default="FedEx", example="FedEx")
    Order_Date: str = Field(default="2024-06-01", example="2024-06-01")
    Delivery_Date: Optional[str] = Field(default=None, example=None)
    Size_Ordered: Optional[str] = Field(default="M", example="M")
    Bought_Two_Sizes: int = Field(default=0, ge=0, le=1, example=0)
    Review_Score: int = Field(default=3, ge=1, le=5, example=3)
    Return_Reason: str = Field(default="", example="")
    Is_Returned: int = Field(default=0, ge=0, le=1, example=0)


class PredictionResponse(BaseModel):
    """Response payload containing return probability and risk tier."""

    return_probability: float = Field(description="Probability of return (0–1)")
    risk_tier: str = Field(description="LOW | MEDIUM | HIGH")
    recommended_action: str = Field(description="Suggested demarketing action")
    model_version: str = Field(default="xgboost-v1")
