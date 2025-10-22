"""
Historical Sales Data Generator
Generates realistic sales data for different products, plants, and areas
"""

import numpy as np


# Base historical data structure (Million Yen)
HISTORICAL_DATA_BASE = {
    'DNHA-M100': {
        'Tokyo': {'Domestic': [1200, 1250, 1300], 'North America': [800, 850, 900],
                  'Europe': [600, 620, 650], 'Asia Pacific': [900, 950, 1000]},
        'Osaka': {'Domestic': [1100, 1150, 1180], 'North America': [750, 780, 820],
                  'Europe': [550, 570, 600], 'Asia Pacific': [850, 880, 920]},
        'Nagoya': {'Domestic': [1300, 1350, 1400], 'North America': [850, 900, 950],
                   'Europe': [650, 680, 710], 'Asia Pacific': [950, 1000, 1050]},
        'Kyushu': {'Domestic': [1000, 1050, 1100], 'North America': [700, 730, 770],
                   'Europe': [500, 520, 550], 'Asia Pacific': [800, 830, 870]}
    },
    'DNHA-M200': {
        'Tokyo': {'Domestic': [1500, 1580, 1650], 'North America': [1000, 1050, 1100],
                  'Europe': [750, 780, 820], 'Asia Pacific': [1100, 1150, 1200]},
        'Osaka': {'Domestic': [1400, 1470, 1530], 'North America': [950, 990, 1040],
                  'Europe': [700, 730, 770], 'Asia Pacific': [1050, 1100, 1150]},
        'Nagoya': {'Domestic': [1600, 1680, 1750], 'North America': [1050, 1100, 1150],
                   'Europe': [800, 830, 870], 'Asia Pacific': [1150, 1200, 1250]},
        'Kyushu': {'Domestic': [1300, 1370, 1430], 'North America': [900, 940, 990],
                   'Europe': [650, 680, 720], 'Asia Pacific': [1000, 1050, 1100]}
    },
    'DNHA-M300': {
        'Tokyo': {'Domestic': [2000, 2100, 2200], 'North America': [1300, 1400, 1500],
                  'Europe': [1000, 1050, 1100], 'Asia Pacific': [1500, 1580, 1650]},
        'Osaka': {'Domestic': [1900, 1990, 2080], 'North America': [1250, 1330, 1420],
                  'Europe': [950, 1000, 1050], 'Asia Pacific': [1450, 1520, 1590]},
        'Nagoya': {'Domestic': [2100, 2200, 2300], 'North America': [1350, 1450, 1550],
                   'Europe': [1050, 1100, 1150], 'Asia Pacific': [1550, 1630, 1710]},
        'Kyushu': {'Domestic': [1800, 1890, 1980], 'North America': [1200, 1280, 1370],
                   'Europe': [900, 950, 1000], 'Asia Pacific': [1400, 1470, 1540]}
    },
    'DNHA-M400': {
        'Tokyo': {'Domestic': [2500, 2650, 2800], 'North America': [1600, 1700, 1800],
                  'Europe': [1200, 1280, 1350], 'Asia Pacific': [1800, 1900, 2000]},
        'Osaka': {'Domestic': [2400, 2540, 2680], 'North America': [1550, 1640, 1730],
                  'Europe': [1150, 1220, 1290], 'Asia Pacific': [1750, 1840, 1930]},
        'Nagoya': {'Domestic': [2600, 2750, 2900], 'North America': [1650, 1750, 1850],
                   'Europe': [1250, 1330, 1400], 'Asia Pacific': [1850, 1950, 2050]},
        'Kyushu': {'Domestic': [2300, 2440, 2580], 'North America': [1500, 1590, 1680],
                   'Europe': [1100, 1170, 1240], 'Asia Pacific': [1700, 1790, 1880]}
    }
}


def generate_historical_data(product="All Products", plant="All Plants", area="All Areas", granularity="annual"):
    """
    Generate aggregated historical sales data based on filters

    Parameters:
    -----------
    product : str
        Product category (e.g., "DNHA-M100", "All Products")
    plant : str
        Plant location (e.g., "Tokyo", "All Plants")
    area : str
        Sales area (e.g., "Domestic", "All Areas")
    granularity : str
        Data granularity: "annual" (3 points), "quarterly" (12 points), or "monthly" (36 points)

    Returns:
    --------
    list : Aggregated historical sales data
    """

    aggregated = [0, 0, 0]  # FY23, FY24, FY25

    # Determine which products to include
    if product == "All Products":
        products = list(HISTORICAL_DATA_BASE.keys())
    else:
        products = [product]

    # Aggregate data
    for prod in products:
        if prod not in HISTORICAL_DATA_BASE:
            continue

        # Determine which plants to include
        if plant == "All Plants":
            plants = list(HISTORICAL_DATA_BASE[prod].keys())
        else:
            plants = [plant]

        for plt in plants:
            if plt not in HISTORICAL_DATA_BASE[prod]:
                continue

            # Determine which areas to include
            if area == "All Areas":
                areas = list(HISTORICAL_DATA_BASE[prod][plt].keys())
            else:
                areas = [area]

            for ar in areas:
                if ar not in HISTORICAL_DATA_BASE[prod][plt]:
                    continue

                data = HISTORICAL_DATA_BASE[prod][plt][ar]
                aggregated = [aggregated[i] + data[i] for i in range(3)]

    # Interpolate to create more granular data if requested
    if granularity == "quarterly":
        return _interpolate_quarterly(aggregated)
    elif granularity == "monthly":
        return _interpolate_monthly(aggregated)

    return aggregated


def _interpolate_quarterly(annual_data):
    """
    Interpolate annual data to quarterly data with seasonal variation

    Parameters:
    -----------
    annual_data : list
        Annual sales data [FY23, FY24, FY25]

    Returns:
    --------
    list : Quarterly sales data (12 points = 3 years * 4 quarters)
    """
    quarterly_data = []

    # Seasonal multipliers for each quarter (Q1, Q2, Q3, Q4)
    # Based on typical automotive industry patterns
    seasonal_pattern = [0.90, 0.95, 1.05, 1.10]  # Sum = 4.0

    for i in range(len(annual_data)):
        annual_value = annual_data[i]
        base_quarterly = annual_value / 4.0

        # Add seasonal variation and small random noise
        for quarter_idx in range(4):
            seasonal_value = base_quarterly * seasonal_pattern[quarter_idx]
            # Add small noise (±2%) for realism
            noise = np.random.uniform(-0.02, 0.02) * seasonal_value
            quarterly_data.append(seasonal_value + noise)

    return quarterly_data


def _interpolate_monthly(annual_data):
    """
    Interpolate annual data to monthly data with seasonal variation

    Parameters:
    -----------
    annual_data : list
        Annual sales data [FY23, FY24, FY25]

    Returns:
    --------
    list : Monthly sales data (36 points = 3 years * 12 months)
    """
    monthly_data = []

    # Seasonal multipliers for each month (12 months)
    # Based on typical automotive industry patterns
    seasonal_pattern = [
        0.85, 0.88, 0.92,  # Q1: Jan, Feb, Mar
        0.94, 0.96, 0.98,  # Q2: Apr, May, Jun
        1.02, 1.05, 1.08,  # Q3: Jul, Aug, Sep
        1.10, 1.12, 1.10   # Q4: Oct, Nov, Dec
    ]  # Sum = 12.0

    for i in range(len(annual_data)):
        annual_value = annual_data[i]
        base_monthly = annual_value / 12.0

        # Add seasonal variation and small random noise
        for month_idx in range(12):
            seasonal_value = base_monthly * seasonal_pattern[month_idx]
            # Add small noise (±1.5%) for realism
            noise = np.random.uniform(-0.015, 0.015) * seasonal_value
            monthly_data.append(seasonal_value + noise)

    return monthly_data


def get_product_info(product):
    """Get information about a product category"""
    product_info = {
        'DNHA-M100': {'description': 'Entry-level model', 'growth_rate': 0.04},
        'DNHA-M200': {'description': 'Mid-range model', 'growth_rate': 0.05},
        'DNHA-M300': {'description': 'Premium model', 'growth_rate': 0.06},
        'DNHA-M400': {'description': 'High-end model', 'growth_rate': 0.07},
        'All Products': {'description': 'All product categories', 'growth_rate': 0.055}
    }
    return product_info.get(product, product_info['All Products'])


def get_plant_info(plant):
    """Get information about a plant location"""
    plant_info = {
        'Tokyo': {'capacity': 'High', 'efficiency': 'Excellent'},
        'Osaka': {'capacity': 'Medium-High', 'efficiency': 'Very Good'},
        'Nagoya': {'capacity': 'Very High', 'efficiency': 'Excellent'},
        'Kyushu': {'capacity': 'Medium', 'efficiency': 'Good'},
        'All Plants': {'capacity': 'Total', 'efficiency': 'Combined'}
    }
    return plant_info.get(plant, plant_info['All Plants'])


def get_area_info(area):
    """Get information about a sales area"""
    area_info = {
        'Domestic': {'market_size': 'Large', 'growth_potential': 'Stable'},
        'North America': {'market_size': 'Large', 'growth_potential': 'High'},
        'Europe': {'market_size': 'Medium', 'growth_potential': 'Moderate'},
        'Asia Pacific': {'market_size': 'Medium', 'growth_potential': 'Very High'},
        'All Areas': {'market_size': 'Global', 'growth_potential': 'Varied'}
    }
    return area_info.get(area, area_info['All Areas'])
