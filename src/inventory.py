"""
RetailPulse Forecast - Supply Chain & Inventory Planning Policy Module
Source: Silver-Pyke-Peterson Inventory & Operations Research Framework

Calculates mathematically grounded safety stock buffers and dynamic reorder points (ROP):
- Lead-Time Uncertainty: sigma_L = sigma_weekly * sqrt(L)
- Safety Stock: SS = Z_sla * sigma_weekly * sqrt(L)
- Reorder Point: ROP = (Weekly_Demand * L) + SS
"""

import numpy as np
from scipy import stats
from typing import Dict, Any

def calculate_inventory_policy(
    weekly_demand: float,
    lead_time_weeks: float,
    uncertainty_sigma: float,
    service_level: float = 0.95
) -> Dict[str, float]:
    """
    Computes statistically rigorous inventory parameters for retail demand planning.

    Parameters
    ----------
    weekly_demand : float
        Expected weekly demand volume or value ($ USD). Must be >= 0.
    lead_time_weeks : float
        Supplier replenishment lead time in weeks (L >= 0).
    uncertainty_sigma : float
        Standard deviation of weekly forecast residuals (or demand variance) ($ USD). Must be >= 0.
    service_level : float, default=0.95
        Target cycle service level (in-stock probability during lead time). Valid range: (0.50, 0.999).

    Returns
    -------
    dict
        Dictionary containing:
        - 'service_level': float (e.g. 0.95)
        - 'z_score': float (e.g. 1.645)
        - 'lead_time_weeks': float
        - 'weekly_demand': float
        - 'cycle_demand': float (expected demand over lead time)
        - 'safety_stock': float (buffer against forecast variance)
        - 'reorder_point': float (cycle demand + safety stock)
    """
    # Defensive validations
    weekly_demand = max(0.0, float(weekly_demand))
    lead_time_weeks = max(0.0, float(lead_time_weeks))
    uncertainty_sigma = max(0.0, float(uncertainty_sigma))
    
    if not (0.50 <= service_level < 1.0):
        raise ValueError(f"Service level must be between 0.50 and 0.999, got {service_level}")

    # Standard normal critical value for target service level SLA
    z_score = float(stats.norm.ppf(service_level))
    
    # Expected demand during replenishment lead time (DDLT)
    cycle_demand = weekly_demand * lead_time_weeks
    
    # Lead-time uncertainty scaling under independent weekly errors: sigma_L = sigma * sqrt(L)
    if lead_time_weeks > 0 and uncertainty_sigma > 0:
        lead_time_sigma = uncertainty_sigma * np.sqrt(lead_time_weeks)
        safety_stock = z_score * lead_time_sigma
    else:
        safety_stock = 0.0
        
    reorder_point = cycle_demand + safety_stock
    
    return {
        "service_level": round(service_level, 4),
        "z_score": round(z_score, 4),
        "lead_time_weeks": round(lead_time_weeks, 2),
        "weekly_demand": round(weekly_demand, 2),
        "cycle_demand": round(cycle_demand, 2),
        "safety_stock": round(safety_stock, 2),
        "reorder_point": round(reorder_point, 2)
    }
