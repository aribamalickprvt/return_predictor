"""
Synthetic E-Commerce Dataset Generator
Generates realistic transaction data with complex patterns for return prediction.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

N_CUSTOMERS = 2000
N_ORDERS = 15000

CATEGORIES = ["Clothing", "Electronics", "Footwear", "Home & Kitchen", "Books", "Toys", "Beauty"]
SHIPPING_METHODS = ["Express", "Standard", "Economy", "Next Day"]
CARRIERS = ["FedEx", "UPS", "USPS", "DHL", "BlueDart"]
SIZES = ["XS", "S", "M", "L", "XL", "XXL", None]  # None for non-clothing

RETURN_REASONS = [
    "Wrong size", "Defective product", "Not as described",
    "Changed mind", "Received wrong item", "Quality issues", ""
]


def generate_customers(n: int) -> pd.DataFrame:
    customer_ids = [f"CUST_{i:05d}" for i in range(1, n + 1)]
    zip_codes = [f"{random.randint(10000, 99999)}" for _ in range(n)]
    account_age_days = np.random.randint(30, 2000, size=n)
    past_purchases = np.random.randint(1, 100, size=n)
    past_returns = np.array([
        np.random.randint(0, max(1, int(p * 0.3))) for p in past_purchases
    ])
    return pd.DataFrame({
        "Customer_ID": customer_ids,
        "ZIP_Code": zip_codes,
        "Account_Age_Days": account_age_days,
        "Past_Purchases": past_purchases,
        "Past_Returns": past_returns,
    })


def generate_orders(customers_df: pd.DataFrame, n: int) -> pd.DataFrame:
    rows = []
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2024, 6, 1)

    for i in range(n):
        cust = customers_df.sample(1).iloc[0]
        category = random.choice(CATEGORIES)
        order_date = start_date + timedelta(
            days=random.randint(0, (end_date - start_date).days)
        )

        # Carrier affects delivery time
        carrier = random.choice(CARRIERS)
        shipping = random.choice(SHIPPING_METHODS)
        base_days = {"Express": 2, "Standard": 5, "Economy": 8, "Next Day": 1}[shipping]
        carrier_delay = {"FedEx": 0, "UPS": 0, "USPS": 1, "DHL": 0, "BlueDart": 2}[carrier]

        # Introduce some missing delivery dates (~5%)
        delivery_date = None
        if random.random() > 0.05:
            delivery_date = order_date + timedelta(
                days=base_days + carrier_delay + random.randint(0, 2)
            )

        price = round(np.random.lognormal(mean=3.5, sigma=1.0), 2)
        price = max(5.0, min(price, 3000.0))

        discount = round(random.choice([0, 0, 0, 5, 10, 15, 20, 25, 30, 50]), 2)

        # Size feature: only clothing/footwear get sizes
        size_ordered = None
        bought_two_sizes = False
        if category in ["Clothing", "Footwear"]:
            size_ordered = random.choice(SIZES[:-1])  # exclude None
            # ~8% chance customer bought two sizes (size mismatch risk)
            bought_two_sizes = random.random() < 0.08

        product_id = f"PROD_{random.randint(1000, 9999)}"
        review_score = random.choices([1, 2, 3, 4, 5], weights=[5, 10, 20, 35, 30])[0]

        # --- Return probability logic ---
        return_prob = 0.10  # base rate

        # Customer return history
        return_ratio = cust["Past_Returns"] / max(cust["Past_Purchases"], 1)
        return_prob += return_ratio * 0.35

        # Size mismatch — very strong signal
        if bought_two_sizes:
            return_prob += 0.45

        # Clothing is returned more
        if category == "Clothing":
            return_prob += 0.12
        elif category == "Electronics":
            return_prob += 0.08

        # High discount impulse buys returned more
        if discount >= 30:
            return_prob += 0.10

        # Low review → more returns
        if review_score <= 2:
            return_prob += 0.20

        # Long delivery → more returns
        if delivery_date and (delivery_date - order_date).days > 7:
            return_prob += 0.08

        # High price items returned less (deliberate buys)
        if price > 500:
            return_prob -= 0.05

        return_prob = max(0.02, min(return_prob, 0.95))
        is_returned = int(random.random() < return_prob)

        return_reason = ""
        if is_returned:
            if bought_two_sizes:
                return_reason = "Wrong size"
            else:
                return_reason = random.choice(RETURN_REASONS[1:])

        rows.append({
            "Order_ID": f"ORD_{i+1:06d}",
            "Customer_ID": cust["Customer_ID"],
            "ZIP_Code": cust["ZIP_Code"],
            "Account_Age_Days": cust["Account_Age_Days"],
            "Past_Purchases": cust["Past_Purchases"],
            "Past_Returns": cust["Past_Returns"],
            "Product_ID": product_id,
            "Category": category,
            "Price": price,
            "Discount_Percentage": discount,
            "Shipping_Method": shipping,
            "Carrier": carrier,
            "Order_Date": order_date.strftime("%Y-%m-%d"),
            "Delivery_Date": delivery_date.strftime("%Y-%m-%d") if delivery_date else None,
            "Size_Ordered": size_ordered,
            "Bought_Two_Sizes": int(bought_two_sizes),
            "Review_Score": review_score,
            "Return_Reason": return_reason,
            "Is_Returned": is_returned,
        })

    return pd.DataFrame(rows)


def main():
    print("Generating synthetic e-commerce dataset...")
    customers = generate_customers(N_CUSTOMERS)
    orders = generate_orders(customers, N_ORDERS)

    output_path = "/return_predictor/data/ecommerce_returns.csv"
    orders.to_csv(output_path, index=False)

    print(f"Dataset saved: {output_path}")
    print(f"Shape: {orders.shape}")
    print(f"Return rate: {orders['Is_Returned'].mean():.2%}")
    print(orders.head(3).to_string())


if __name__ == "__main__":
    main()
