# generate_sample_data.py
"""
Generate sample house price data for demonstration purposes.
This creates a CSV file with realistic-looking house data.

Usage:
    python generate_sample_data.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

def generate_sample_data(n_samples=1000):
    """Generate synthetic house price data"""
    
    # Locations (Mumbai area examples)
    locations = ['Thane', 'Andheri', 'Bandra', 'Powai', 'Borivali', 
                 'Malad', 'Goregaon', 'Kandivali', 'Mulund', 'Ghatkopar']
    
    # Status options
    status_options = ['Ready to move', 'Under construction', 'New launch']
    
    # Transaction types
    transaction_types = ['New', 'Resale']
    
    # Furnishing types
    furnishing_types = ['Furnished', 'Semi-Furnished', 'Unfurnished']
    
    # Facing directions
    facing_options = ['North', 'South', 'East', 'West', 'North-East', 
                      'North-West', 'South-East', 'South-West']
    
    # Overlooking options
    overlooking_options = ['Park', 'Main Road', 'Garden', 'Pool', 'City', 'Club']
    
    # Ownership types
    ownership_types = ['Freehold', 'Leasehold', 'Co-operative society', 'Power of Attorney']
    
    data = []
    
    for i in range(n_samples):
        # Generate correlated features
        carpet_area = np.random.randint(400, 1500)
        super_area = carpet_area + np.random.randint(100, 400)
        
        floor = np.random.randint(0, 25)
        bathroom = min(np.random.randint(1, 5), carpet_area // 300 + 1)
        balcony = np.random.randint(0, 4)
        parking = np.random.randint(0, 3)
        
        location = np.random.choice(locations)
        
        # Base price calculation (simplified)
        # Price influenced by area, location, floor, etc.
        base_price = carpet_area * np.random.uniform(8000, 15000)
        
        # Location multiplier
        location_multipliers = {
            'Bandra': 1.8, 'Andheri': 1.4, 'Powai': 1.5,
            'Thane': 1.0, 'Borivali': 1.1, 'Malad': 1.15,
            'Goregaon': 1.2, 'Kandivali': 1.1, 'Mulund': 1.05,
            'Ghatkopar': 1.15
        }
        base_price *= location_multipliers.get(location, 1.0)
        
        # Add some randomness
        price = base_price * np.random.uniform(0.85, 1.15)
        
        # Sometimes express as amount (could be down payment or different context)
        # Make it 20-40% of price to avoid perfect correlation
        amount = price * np.random.uniform(0.2, 0.4)
        
        row = {
            'Index': i + 1,
            'Title': f'{carpet_area} sqft apartment in {location}',
            'Description': f'Spacious {bathroom}BHK apartment with {balcony} balconies',
            'Amount(in rupees)': amount,
            'Carpet Area': carpet_area,
            'Floor': floor,
            'Bathroom': bathroom,
            'Balcony': balcony,
            'Car Parking': parking,
            'Super Area': super_area,
            'location': location,
            'Status': np.random.choice(status_options),
            'Transaction': np.random.choice(transaction_types),
            'Furnishing': np.random.choice(furnishing_types),
            'facing': np.random.choice(facing_options),
            'overlooking': np.random.choice(overlooking_options),
            'Ownership': np.random.choice(ownership_types),
            'Society': f'{location} Heights' if i % 3 == 0 else '',
            'Dimensions': f'{carpet_area} sq.ft.' if i % 2 == 0 else '',
            'Plot Area': '' if i % 5 != 0 else str(np.random.randint(1000, 3000)),
            'Price (in rupees)': price
        }
        
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Add some missing values to make it realistic
    missing_rate = 0.05
    for col in df.columns:
        if col not in ['Index', 'Price (in rupees)']:
            mask = np.random.random(len(df)) < missing_rate
            df.loc[mask, col] = np.nan
    
    return df

if __name__ == '__main__':
    print("Generating sample house price data...")
    df = generate_sample_data(n_samples=1000)
    
    output_file = 'house_prices.csv'
    df.to_csv(output_file, index=False)
    
    print(f"✓ Generated {len(df)} records")
    print(f"✓ Saved to {output_file}")
    print(f"\nDataset shape: {df.shape}")
    print(f"\nSample statistics:")
    print(df[['Carpet Area', 'Floor', 'Bathroom', 'Price (in rupees)']].describe())
    print(f"\nMissing values:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
