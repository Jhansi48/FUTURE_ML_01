# Dataset Documentation: Retail Multi-Store Sales

## Source & Description
This dataset captures weekly historical department sales across multiple retail store formats from **January 2021 through June 2024** (5,490 aggregate records across 5 stores and 6 departments).

Modeled on benchmark retail operational time series (Walmart / Superstore benchmark), it encompasses:
- `Date`: Weekly Friday timestamp
- `Store_ID`: Store identifier (1 to 5)
- `Store_Type`: Store tier (A: Flagship, B: Standard, C: Compact)
- `Store_Size`: Square footage (50,000 to 150,000 sq ft)
- `Dept_ID` & `Dept_Name`: Departments (Electronics, Apparel, Grocery, Home & Garden, Toys, Health & Beauty)
- `Weekly_Sales`: Target variable ($ USD)
- `IsHoliday` & `Holiday_Name`: Key holiday weeks (Super Bowl, Memorial Day, Labor Day, Thanksgiving/Black Friday, Christmas/New Year)
- `Markdown_Discount` & `Promotional_Flag`: Promotional marketing expenditures
- `Temperature`, `Fuel_Price`, `CPI`, `Unemployment_Rate`: External economic indicators

## Data Integrity & Chronology
- No look-ahead leakage.
- Strict chronological order preserved for time-series modeling.
