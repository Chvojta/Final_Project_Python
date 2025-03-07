import os
import numpy as np
import pandas as pd
from benchmarks import CLOUD_SAVINGS_BENCHMARKS, OTHER_COGS_BENCHMARKS, salary_benchmarks

"""
This is a small function that calculates potential R&D savings based on the average compensation of the employee and the benchmark compensation for that role. It takes in a row of the dataframe for the R&D employee data and returns a series of the savings for each strategy, by finding the delta between the average comp and the benchamark comp, and multilpying through by the number of relevant employees.
From there, it returns a series of the savings for each strategy with a key value pair of the savings labelled as Savings_High, Savings_Medium, or Savings_Low, based on the strategy. If there's no title data for that row, it jsut returns 0s for that row across all strategies.
"""
def calculate_savings(row):
    title = row["Title"]
    avg_comp = row["Average_Comp"]
    emp_count = row["Employee_Count"]

    if title in salary_benchmarks:
        benchmarks = salary_benchmarks[title]
 
        return pd.Series({
            # We use Max here to protect form potentially negative savings (if the avg comp is lower than the benchmark)
            "Savings_High": max(0, (avg_comp - benchmarks["high"])) * emp_count,
            "Savings_Medium": max(0, (avg_comp - benchmarks["medium"])) * emp_count,
            "Savings_Low": max(0, (avg_comp - benchmarks["low"])) * emp_count
        })
    else:
        return pd.Series({"Savings_High": 0, "Savings_Medium": 0, "Savings_Low": 0})

"""
This is the main function that calculates all the metrics for the savings analysis. It takes in a strategy, a list of selected analyses, and a dataframe of input data. It then calculates all the metrics for the savings analysis and returns a dictionary of the metrics.
It also has a breakdown of the ebitda, which is used to calculate the savings for each analysis.
Further documentation is below.
The key return value is the metrics dictionary. In the end, it's structure is as follows, and the values used for the following sections:
        dict: A metrics dictionary with the following structure:
            - 'Revenue': {
                'original': numpy.ndarray[3] - the original revenue for 3 years
                'improved': numpy.ndarray[3], - Improved revenue with all selected analyses
                'sales_improved': numpy.ndarray[3], - Revenue with only sales improvements
                'cam_improved': numpy.ndarray[3],  - Revenue with only CAM improvements
                'support_improved': numpy.ndarray[3] - Revenue with only support improvements
            }
            - 'Growth_Rate': {
                'original': numpy.ndarray[3],  - Original growth rates (%)
                'improved': numpy.ndarray[3]   - Improved growth rates (%)
            }
            - 'EBITDA': {
                'original': numpy.ndarray[3], - Original EBITDA values from the income statement
                'improved': numpy.ndarray[3]  - Improved EBITDA values with all analyses, for the 3 years 
            }
            - 'Margin': {
                'original': numpy.ndarray[3], - Original EBITDA margin percentages, to show in results template
                'improved': numpy.ndarray[3]  -  Improved EBITDA margin percentages, to show in results template
            }
            - 'Rule_Of': {
                'original': numpy.ndarray[3], - Original Rule of values, to show in results template
                'improved': numpy.ndarray[3]  - Improved Rule of values, to show in results template
            }
            - 'EBITDA_Multiple': {
                'original': numpy.ndarray[3], - Original EBITDA multiples, to show in results template
                'improved': numpy.ndarray[3]  - Improved EBITDA multiples, to show in results template
            }
            - 'Enterprise_Value': {
                'original': numpy.ndarray[3], - Original enterprise values, to show in results template
                'improved': numpy.ndarray[3]  - Improved enterprise values, to show in results template
            }
            - 'Years': list[3], - List of years [2022, 2023, 2024]
            - 'Row_Labels': list[7], - List of row labels for display
            
            # Analysis-specific metrics (included only when those analyses are selected):
            - 'ebitda_breakdown': dict, # Maps analysis names to their EBITDA contribution
            - 'rd_analysis': {
                'roles': list, # Job titles
                'employee_counts': list, # Number of employees per role 
                'current_comp': list, # Current compensation per role
                'benchmark_comp': list, # Benchmark compensation per role
                'role_savings': list, # Savings per role
                'total_savings': float # Total R&D savings
            }
            - 'support_analysis': {
                'potential_support_revenue': list[3], # Support revenue by year
                'annual_revenue': float, # Annual support revenue (year 3)
                'premium_fee': float, # Premium support fee per customer
                'chart_data': dict, # Data for support charts
                'top_customers': list, # List of top customer IDs (if available)
                'customer_revenue': list # Revenue per top customer (if available)
            }
            - 'sales_analysis': {
                'attainment_buckets': list, # Sales rep attainment bucket labels 
                'rep_counts': list, # Number of reps in each attainment bucket
                'current_revenue': list, # Current revenue by year
                'improved_revenue': list # Improved revenue by year with sales optimization
            }
            - 'cam_analysis': {
                'attainment_buckets': list, # CAM rep attainment bucket labels
                'rep_counts': list, # Number of CAMs in each attainment bucket
                'current_revenue': list, # Current revenue by year
                'improved_revenue': list # Improved revenue by year with CAM optimization
            }
            - 'other_cogs_analysis': {
                'current_pct': numpy.ndarray[3], # Current other COGS as % of revenue
                'benchmark_pct': numpy.ndarray[3], # Benchmark COGS % targets
                'potential_savings': numpy.ndarray[3], # Potential savings by year
                'is_above_benchmark': numpy.ndarray[3] # Boolean flags if current % > benchmark
            }
            - 'cloud_analysis': {
                'current_pct': numpy.ndarray[3], # Current cloud costs as % of revenue
                'benchmark_pct': numpy.ndarray[3], # Benchmark cloud % targets
                'total_savings': int, # Total cloud savings
                'current_costs': numpy.ndarray[3], # Current cloud costs
                'benchmark_costs': numpy.ndarray[3] # Target cloud costs based on benchmarks
            }
            
            # Data arrays returned to the template:
            - 'cloud_savings_original': numpy.ndarray[3], provides thhe cloud savings / yr
            - 'other_cogs_savings_original': numpy.ndarray[3], - provides the  COGS savings / yr
            - 'rd_savings': float, # Total R&D savings (single value) for 2024
            - 'quota_improvement': numpy.ndarray[3], combined quota improvements per year
            - 'sales_quota_improvement': numpy.ndarray[3], - sales quota improvements by year
            - 'cam_quota_improvement': numpy.ndarray[3],  - cAM quota improvements by year
            - 'support_revenue': numpy.ndarray[3], - support revenue by year
            - 'selected_analyses': list  - the list of selected analysis names from the user
"""

def calculate_all_metrics(strategy="Medium", selected_analyses=None, df_inputs=None):   
    # We start by initializing an empty metrics dictionary, a running total of savings, and create teh ebitda breakdown dictionary we will use later on to show each analysis's contribution total savings.
    metrics = {}
    running_total = 0
    metrics['ebitda_breakdown'] = {}
    # Extract income statement for calculations
    income_statement = df_inputs.get('Income Statement', pd.DataFrame())
    
    # Extract original revenue and EBITDA for baseline calculations from teh provided data frame from teh excel we uplaod. and flatten the data to make it easier to work with
    if 'Revenue' in income_statement['Line Item'].values and 'EBITDA' in income_statement['Line Item'].values:
        original_revenue = income_statement.loc[income_statement['Line Item'] == 'Revenue'].iloc[:, 1:4].values.flatten()
        original_ebitda = income_statement.loc[income_statement['Line Item'] == 'EBITDA'].iloc[:, 1:4].values.flatten()
    
    # Initialize numpy arrays of size 3 for each analysis's incremental improvements (will map to 2022, 2023, 2024), initially at 0 so even if no savings found we still pass along the correct data structure
    support_revenue = np.zeros(3)  
    rd_savings_array = np.zeros(3)
    cloud_savings = np.zeros(3)
    other_cogs_savings = np.zeros(3)
    quota_improvement_array = np.zeros(3)
    
    # Initialize sales and CAM improvements and rd savings to prevent unbound local variable errors (we'll use these to store the incremental improvements for each year) and will be used to update the arrays above.
    sales_improvement = {year: 0 for year in [2022, 2023, 2024]}
    cam_improvement = {year: 0 for year in [2022, 2023, 2024]}
    rd_savings = 0

    # If the R&D cost optimization analysis is selected, we call the rocess rd data function to take the rd emloyee data, gropu itby title and get the count, and avg comp, and then call teh calculate savings func above to get the savings for each strategy.
    if 'rd_cost_optimization' in selected_analyses:
        def process_rd_data(df, strategy):
            """Process R&D employee data in a single pass."""
            df_grouped = df.groupby("Title").agg(
                Total_Comp_Sum=("Total Comp", "sum"),
                Employee_Count=("Total Comp", "count"),
                Average_Comp=("Total Comp", "mean")
            ).reset_index()
            
            # Apply savings calculation and add as new columns instead of joining
            savings_data = df_grouped.apply(lambda row: calculate_savings(row), axis=1)
            for col in savings_data.columns:
                df_grouped[col] = savings_data[col]
                
            return df_grouped

        # Use the function - note, we are assuming only 2024 data is provided, so we are only using that year for the savings calculation.
        df_grouped = process_rd_data(df_inputs['R&D employee data'], strategy)
        rd_savings = df_grouped[f"Savings_{strategy}"].sum()
        rd_savings_array = np.array([0, 0, rd_savings])
        
        # Format the data for the template as lists for each key to be used in teh front end templates
        metrics['rd_analysis'] = {
            'roles': df_grouped['Title'].tolist(),
            'Employee_Count': df_grouped['Employee_Count'].tolist(),
            'employee_counts': df_grouped['Employee_Count'].tolist(),
            'current_comp': df_grouped['Average_Comp'].tolist(),
            'benchmark_comp': [salary_benchmarks.get(title, {}).get(strategy.lower(), current) 
                              for title, current in zip(df_grouped['Title'], df_grouped['Average_Comp'])],
            'role_savings': df_grouped[f"Savings_{strategy}"].tolist(),
            'total_savings': rd_savings
        }
    else:
        rd_savings_array = np.zeros(len(original_revenue))

    # Create a  quota_improvement dictionary to store the quota improvement (both sales and CAM) for each year
    quota_improvement = {year: 0 for year in [2022, 2023, 2024]}
    
    # Process support monetization if the analysis is selected, using the strategy and the df_inputs dataframe
    if 'support_monetization' in selected_analyses:
        # Calculate potential support revenue using the calculate_support_revenue function
        support_revenue = calculate_support_revenue(strategy, df_inputs)
        
        if not isinstance(support_revenue, np.ndarray):
            # If it's a single value, instead of an arrray, convert to array with phased implementation
            if support_revenue == 0:
                support_revenue = np.zeros(3)
            else:
                annual_revenue = float(support_revenue)
                support_revenue = np.array([
                    annual_revenue * 0.33,  # implementation schedule
                    annual_revenue * 0.66,  
                    annual_revenue          
                ])
        
        # Ensure the array has length 3 (one for each year)
        if len(support_revenue) != 3:
            # If it's shorter, pad with zeros; if longer, shorten it to 3
            temp_array = np.zeros(3)
            for i in range(min(len(support_revenue), 3)):
                temp_array[i] = support_revenue[i]
            support_revenue = temp_array        
        # Add support_analysis to metrics for the template only if there are non-zero values (otherwise not applicable)
        if np.sum(support_revenue) > 0:
            metrics['support_analysis'] = {
                'potential_support_revenue': support_revenue.tolist(),
                'annual_revenue': float(support_revenue[2]),
                'premium_fee': 3000,
                'chart_data': {
                    'labels': ['Current Support Customers', 'Premium Support Customers'],
                    'values': [0, int(support_revenue[2] / 3000)]  # Calculate number of premium customers
                }
            }
            
            # Try to extract actual customer data if available
            if 'Support Monetization' in df_inputs:
                df_support = df_inputs['Support Monetization']
                if len(df_support) > 0:
                    # Get top 10 customers by ARR
                    if '2024 ARR' in df_support.columns and 'Customer ID' in df_support.columns:
                        top_customers = df_support.sort_values(by='2024 ARR', ascending=False).head(10)
                        metrics['support_analysis']['top_customers'] = top_customers['Customer ID'].tolist()
                        metrics['support_analysis']['customer_revenue'] = top_customers['2024 ARR'].tolist()
        else:
            # If no support revenue found, add empty support_analysis with zeros
            metrics['support_analysis'] = {
                'potential_support_revenue': [0, 0, 0],
                'annual_revenue': 0,
                'premium_fee': 3000,
                'chart_data': {
                    'labels': ['Current Support Customers', 'Premium Support Customers'],
                    'values': [0, 0]
                },
                'top_customers': [],
                'customer_revenue': []
            }
        
        # Store the support revenue contribution in the EBITDA breakdown key for the template
        metrics['ebitda_breakdown']['support_monetization'] = np.sum(support_revenue)
        
        if np.sum(support_revenue) != 0:
            running_total += np.sum(support_revenue)

    # Only process sales/CAM data if those analyses are selected
    if ('sales_optimization' in selected_analyses or 'cam_optimization' in selected_analyses):
        # Initialize quota improvement and initial quota arrays to store the quota improvement and initial quota for each year
        quota_improvement = {year: 0 for year in [2022, 2023, 2024]}
        initial_quota = np.zeros(3)  # Array for 2022, 2023, 2024
        
        # Process sales rep data if the analysis is selected and data exists, otherwise maintain the same structure with 0's if no data available
        if 'sales_optimization' in selected_analyses and 'Sales Rep Data' in df_inputs:
            df = df_inputs['Sales Rep Data']
            for year_idx, year in enumerate([2022, 2023, 2024]):
                initial_quota[year_idx] = df[f'{year} Quota'].sum() if f'{year} Quota' in df.columns else 0
            
            # Iterate through the dataframe and calculate the quota improvement for each year. Note, the expecation is 80% quota attainment, as is market standard. this is applied at a rep level (if an individual rep is below 80% attainment, we calculate the difference and add it to the quota improvement for that year), not the aggregate level.
            for _, row in df.iterrows():
                for year in quota_improvement:
                    quota = row.get(f'{year} Quota', 0)
                    attainment = row.get(f'{year} Attainment', 0)
                    if attainment < 0.8 * quota:
                        improvement = (0.8 * quota) - attainment
                        quota_improvement[year] += improvement
                        sales_improvement[year] += improvement
            
            # Create a sales_analysis dictionary for the template
            attainment_buckets = ["<60%", "60-70%", "70-80%", "80-90%", "90-100%", ">100%"]
            rep_counts = [0, 0, 0, 0, 0, 0]  # Default counts for each bucket
            
            # Count reps in each attainment bucket for the latest year with data
            latest_year = max([2022, 2023, 2024], key=lambda y: df[f'{y} Quota'].sum() if f'{y} Quota' in df.columns and f'{y} Attainment' in df.columns else 0)
            
            if f'{latest_year} Attainment' in df.columns:
                for _, row in df.iterrows():
                    attainment = row.get(f'{latest_year} Attainment', 0)
                    quota = row.get(f'{latest_year} Quota', 0)
                    if quota > 0:  # Only consider reps with quotas
                        pct = attainment / quota * 100
                        if pct < 60:
                            rep_counts[0] += 1
                        elif pct < 70:
                            rep_counts[1] += 1
                        elif pct < 80:
                            rep_counts[2] += 1
                        elif pct < 90:
                            rep_counts[3] += 1
                        elif pct < 100:
                            rep_counts[4] += 1
                        else:
                            rep_counts[5] += 1
            
            # Add to metrics - note, we are using the original revenue and the sales_quota_improvement array to calculate the improved revenue.
            metrics['sales_analysis'] = {
                'attainment_buckets': attainment_buckets,
                'rep_counts': rep_counts,
                'current_revenue': original_revenue.tolist(),
                'improved_revenue': (original_revenue + (sales_quota_improvement if 'sales_quota_improvement' in locals() else np.zeros(3))).tolist()
            }
        else:
            # maintain the same structure with 0's if no data available
            metrics['sales_analysis'] = {
                'attainment_buckets': ["<60%", "60-70%", "70-80%", "80-90%", "90-100%", ">100%"],
                'rep_counts': [0, 0, 0, 0, 0, 0],  # Use zeros instead of fake distribution
                'current_revenue': original_revenue.tolist(),
                'improved_revenue': original_revenue.tolist()  # No improvement
            }
        
        # Process CAM rep data if the analysis is selected and data exists
        if 'cam_optimization' in selected_analyses and 'CAM Rep Data' in df_inputs:
            df = df_inputs['CAM Rep Data']
            # Add CAM quotas to initial quota array for each year
            for year_idx, year in enumerate([2022, 2023, 2024]):
                if f'{year} Quota' in df.columns:
                    initial_quota[year_idx] += df[f'{year} Quota'].sum()

            #SAme deal here - per rep up to 80% attainment, if below, calculate the difference and add it to the quota improvement for that year.
            for _, row in df.iterrows():
                for year in quota_improvement:
                    quota = row.get(f'{year} Quota', 0)
                    attainment = row.get(f'{year} Attainment', 0)
                    if attainment < 0.8 * quota:
                        improvement = (0.8 * quota) - attainment
                        quota_improvement[year] += improvement
                        cam_improvement[year] += improvement
            
            # Create a cam_analysis dictionary for the template
            attainment_buckets = ["<60%", "60-70%", "70-80%", "80-90%", "90-100%", ">100%"]
            rep_counts = [0, 0, 0, 0, 0, 0]  # Default counts for each bucket
            
            # Count reps in each attainment bucket for the latest year with data
            latest_year = max([2022, 2023, 2024], key=lambda y: df[f'{y} Quota'].sum() if f'{y} Quota' in df.columns and f'{y} Attainment' in df.columns else 0)
            
            if f'{latest_year} Attainment' in df.columns:
                for _, row in df.iterrows():
                    attainment = row.get(f'{latest_year} Attainment', 0)
                    quota = row.get(f'{latest_year} Quota', 0)
                    if quota > 0:  # Only consider reps with quotas
                        pct = attainment / quota * 100
                        if pct < 60:
                            rep_counts[0] += 1
                        elif pct < 70:
                            rep_counts[1] += 1
                        elif pct < 80:
                            rep_counts[2] += 1
                        elif pct < 90:
                            rep_counts[3] += 1
                        elif pct < 100:
                            rep_counts[4] += 1
                        else:
                            rep_counts[5] += 1
            
            # Add to metrics - note, we are using the original revenue and the cam_quota_improvement array to calculate the improved revenue.
            metrics['cam_analysis'] = {
                'attainment_buckets': attainment_buckets,
                'rep_counts': rep_counts,
                'current_revenue': original_revenue.tolist(),
                'improved_revenue': (original_revenue + (cam_quota_improvement if 'cam_quota_improvement' in locals() else np.zeros(3))).tolist()
            }
        else:
            # maintain the same structure with 0's if no data available
            metrics['cam_analysis'] = {
                'attainment_buckets': ["<60%", "60-70%", "70-80%", "80-90%", "90-100%", ">100%"],
                'rep_counts': [0, 0, 0, 0, 0, 0],  # Use zeros instead of fake distribution
                'current_revenue': original_revenue.tolist(),
                'improved_revenue': original_revenue.tolist()  # No improvement
            }
    
    # Convert quota_improvement dictionary to array for each year
    quota_improvement_array = np.array([quota_improvement.get(2022, 0), 
                                         quota_improvement.get(2023, 0), 
                                         quota_improvement.get(2024, 0)])
    
    # Convert sales and CAM improvements to arrays as well
    sales_quota_improvement = np.array([sales_improvement.get(2022, 0),
                                        sales_improvement.get(2023, 0),
                                        sales_improvement.get(2024, 0)])
    
    cam_quota_improvement = np.array([cam_improvement.get(2022, 0),
                                     cam_improvement.get(2023, 0),
                                     cam_improvement.get(2024, 0)])
    
    
    # Store cloud analysis data
    if 'cloud_optimization' in selected_analyses:
        # Use the 2024 cloud savings value (index 2) for consistency with other analyses
        if 'cloud_savings_original' in metrics:
            cloud_optimization = metrics['cloud_savings_original'][2]  # Use only 2024 value
        elif 'cloud_analysis' in metrics and 'total_savings' in metrics['cloud_analysis']:
            cloud_optimization = metrics['cloud_analysis']['total_savings']
        else:
            cloud_optimization = 0
            
        # Add to the breakdown for future reference
        metrics['ebitda_breakdown']['cloud_optimization'] = cloud_optimization
        
        # Add to the bridge if there's a non-zero value
        if cloud_optimization != 0:
            running_total += cloud_optimization
            ebitda_bridge.append({
                'label': 'Cloud Optimization',
                'value': cloud_optimization,
                'running_total': running_total
            })

    # Process non-cloud COGS if the analysis is selected
    if 'other_cogs_optimization' in selected_analyses:
        try:
            # Check if 'Other' line item exists in income statement
            if 'Other' in income_statement['Line Item'].values:
                current_other_cogs = income_statement.loc[income_statement['Line Item'] == 'Other'].iloc[:, 1:4].values.flatten()
                current_other_pct = current_other_cogs / original_revenue * 100
                
                benchmark_pct = OTHER_COGS_BENCHMARKS[strategy] * 100
                
                # Calculate potential savings - but only if current is higher than benchmark
                potential_savings = np.where(
                    current_other_pct > benchmark_pct,
                    current_other_cogs - (original_revenue * OTHER_COGS_BENCHMARKS[strategy]),
                    0
                )
                
                other_cogs_savings = potential_savings

            else:
                # If the line item doesn't exist, return zeros
                current_other_cogs = np.zeros_like(original_revenue)
                current_other_pct = np.zeros_like(original_revenue)
                other_cogs_savings = np.zeros_like(original_revenue)
                
            # Create other_cogs_analysis dictionary for the template
            metrics['other_cogs_analysis'] = {
                'current_pct': current_other_pct,
                'benchmark_pct': np.full(3, benchmark_pct if 'benchmark_pct' in locals() else OTHER_COGS_BENCHMARKS[strategy] * 100),
                'potential_savings': other_cogs_savings,
                'is_above_benchmark': current_other_pct > (benchmark_pct if 'benchmark_pct' in locals() else OTHER_COGS_BENCHMARKS[strategy] * 100)
            }
        except Exception as e:
            other_cogs_savings = np.zeros(3)
            
            metrics['other_cogs_analysis'] = {
                'current_pct': np.zeros(3),
                'benchmark_pct': np.full(3, OTHER_COGS_BENCHMARKS[strategy] * 100),
                'potential_savings': np.zeros(3),
                'is_above_benchmark': np.array([False, False, False])
            }

    # Calculate improved EBITDA, starting from original and adding in each analysis if its selected / its savings we returned.
    improved_ebitda = original_ebitda.copy()  # Start with a copy of original EBITDA
    
    # Add improvements from each analysis
    if 'rd_cost_optimization' in selected_analyses:
        improved_ebitda += rd_savings_array
    
    if 'cloud_optimization' in selected_analyses:
        improved_ebitda += cloud_savings
    
    if 'other_cogs_optimization' in selected_analyses:
        # Use the original calculation if available
        if 'other_cogs_savings_original' in metrics:
            other_cogs_contribution = np.sum(metrics['other_cogs_savings_original'])
        elif 'other_cogs_savings' in metrics:
            other_cogs_contribution = np.sum(metrics['other_cogs_savings'])
        else:
            other_cogs_contribution = 0
            
        # Add to the breakdown for future reference
        metrics['ebitda_breakdown']['other_cogs_optimization'] = other_cogs_contribution
        
        # Don't try to append to ebitda_bridge here
        if other_cogs_contribution != 0:
            running_total += other_cogs_contribution
    
    if 'support_monetization' in selected_analyses:
        improved_ebitda += support_revenue  # Support revenue flows directly to EBITDA
    
    if 'sales_optimization' in selected_analyses or 'cam_optimization' in selected_analyses:
        # For sales/CAM optimizations, assume 60% of new revenue flows to EBITDA
        improved_ebitda += quota_improvement_array * 0.6
    
    sales_improved_revenue = original_revenue + sales_quota_improvement
    cam_improved_revenue = original_revenue + cam_quota_improvement
    support_improved_revenue = original_revenue + support_revenue
    
    # For total improved revenue, only include the selected analyses
    new_revenue_by_year = original_revenue.copy()
    if 'sales_optimization' in selected_analyses:
        new_revenue_by_year += sales_quota_improvement
    if 'cam_optimization' in selected_analyses:
        new_revenue_by_year += cam_quota_improvement
    if 'support_monetization' in selected_analyses:
        new_revenue_by_year += support_revenue


    # Instead, use the already calculated improved_ebitda
    new_ebitda_by_year = improved_ebitda

    # Calculate Rule of values first, which will both be displayed nad also used to get the new bitda multiple
    rule_of_original = (original_ebitda / original_revenue) * 100 + np.array([0] + [(original_revenue[i] / original_revenue[i-1] - 1) * 100 for i in range(1, 3)])
    rule_of_improved = (new_ebitda_by_year / new_revenue_by_year) * 100 + np.array([0] + [(new_revenue_by_year[i] / new_revenue_by_year[i-1] - 1) * 100 for i in range(1, 3)])
    
    # Calculate EBITDA multiple based on Rule of value with proprietary formula: multiple = 6 + 0.15 * (rule_of - 20) = 3 + 0.15 * rule_of
    
    original_multiple = np.maximum(5.0, 3 + 0.15 * rule_of_original)  # Floor of 5.0x
    improved_multiple = np.maximum(5.0, 3 + 0.15 * rule_of_improved)  # Floor of 5.0x
    original_enterprise_value = original_ebitda * original_multiple
    improved_enterprise_value = new_ebitda_by_year * improved_multiple

    #Create the metrics dictionary with all calculated values
    metrics.update({
        'Revenue': {
            'original': original_revenue,
            'improved': new_revenue_by_year,
            'sales_improved': sales_improved_revenue,
            'cam_improved': cam_improved_revenue,
            'support_improved': support_improved_revenue
        },
        'Growth_Rate': {
            'original': np.array([0] + [(original_revenue[i] / original_revenue[i-1] - 1) * 100 for i in range(1, 3)]),
            'improved': np.array([0] + [(new_revenue_by_year[i] / new_revenue_by_year[i-1] - 1) * 100 for i in range(1, 3)])
        },
        'EBITDA': {
            'original': original_ebitda,
            'improved': new_ebitda_by_year
        },
        'Margin': {
            'original': (original_ebitda / original_revenue) * 100,
            'improved': (new_ebitda_by_year / new_revenue_by_year) * 100
        },
        'Rule_Of': {
            'original': rule_of_original,
            'improved': rule_of_improved
        },
        'EBITDA_Multiple': {
            'original': original_multiple,
            'improved': improved_multiple
        },
        'Enterprise_Value': {
            'original': original_enterprise_value,
            'improved': improved_enterprise_value
        },
        'Years': [2022, 2023, 2024],
        'Row_Labels': ["Revenue", "Growth Rate", "EBITDA", "Margin", "Rule of", "EBITDA Multiple", "Enterprise Value"],
        'cloud_savings_original': cloud_savings,
        'other_cogs_savings_original': other_cogs_savings,
        'rd_savings': rd_savings,
        'quota_improvement': quota_improvement_array,
        'sales_quota_improvement': sales_quota_improvement,
        'cam_quota_improvement': cam_quota_improvement,
        'support_revenue': support_revenue  # Add support_revenue to metrics
    })

    # Add the selected analyses to metrics for template access
    metrics['selected_analyses'] = selected_analyses

    return metrics

"""
This function is used to fomrat data for the results page, and takes in the metrics dictionary and fomrats it in row / column form for teh income statements, with each row labeleld correctly for the html tempalte
"""
def analyze_data(strategy='Medium', selected_analyses=None, df_inputs=None):
    # Get metrics from savings calculations and format it for the results page
    metrics = calculate_all_metrics(strategy, selected_analyses, df_inputs)
    row_labels = ["Revenue", "Growth Rate", "EBITDA", "Margin", "Rule of", "EBITDA Multiple", "Enterprise Value"]
    years = [2022, 2023, 2024]
    
    # Ensure consistent naming between the row_labels and metrics dictionary keys
    label_to_key = {
        "Revenue": "Revenue",
        "Growth Rate": "Growth_Rate",
        "EBITDA": "EBITDA",
        "Margin": "Margin", 
        "Rule of": "Rule_Of",
        "EBITDA Multiple": "EBITDA_Multiple",
        "Enterprise Value": "Enterprise_Value"
    }
    
    original_values = []
    improved_values = []
    
    # Loop through row labels, appending each value to the original and improved income satement arrays
    for label in row_labels:
        key = label_to_key[label]
        original_values.append(metrics[key]['original'])
        improved_values.append(metrics[key]['improved'])
    
    # Return the values needed by template
    return row_labels, years, original_values, improved_values

"""
This function is used to detect which analyses can be performed based on which sheets are present in teh input file template, and also returns a quick blurb about each analysis that the tempalte wil render when teh user is selecting which analyses to run.
"""
def detect_available_analyses(excel_file_path="Inputs.xlsx", df_inputs=None):
    """Detect which analyses can be performed based on available data."""
    available_analyses = {}
    
    try:
        # Try to load all sheets from the Excel file if not provided
        if df_inputs is None:
            df_inputs = pd.read_excel(excel_file_path, sheet_name=None)
        # Core analyses - add these when corresponding data is available
        if 'R&D employee data' in df_inputs:
            available_analyses['rd_cost_optimization'] = {
                'name': 'R&D Labor Cost Optimization',
                'description': 'Optimize R&D costs by comparing against industry benchmarks.'
            }
        
        if 'Support Monetization' in df_inputs:
            available_analyses['support_monetization'] = {
                'name': 'Support Monetization',
                'description': 'Generate additional revenue through premium support offerings.'
            }
        
        if 'Income Statement' in df_inputs:
            df = df_inputs['Income Statement']
            # Check for Hosting Costs
            if any('Hosting Costs' in str(item) for item in df['Line Item'].values if item is not None):
                available_analyses['cloud_optimization'] = {
                    'name': 'Cloud Infrastructure Cost Optimization', 
                    'description': 'Reduce cloud hosting costs based on industry standards.'
                }
            
            # Non-Cloud COGS - separate analysis
            if any('Other' in str(item) for item in df['Line Item'].values if item is not None):
                available_analyses['other_cogs_optimization'] = {
                    'name': 'Non-Cloud COGS Optimization',
                    'description': 'Optimize other cost of goods sold not related to cloud infrastructure.'
                }
        
        # Check for Sales Rep Data
        if 'Sales Rep Data' in df_inputs:
            available_analyses['sales_optimization'] = {
                'name': 'Sales Rep Performance Optimization',
                'description': 'Improve sales rep quota attainment to industry standards.'
            }
        
        # Check for CAM Rep Data
        if 'CAM Rep Data' in df_inputs:
            available_analyses['cam_optimization'] = {
                'name': 'CAM Rep Optimization',
                'description': 'Enhance customer account manager performance and retention metrics.'
            }
        
    except Exception as e:
        print(f"Error detecting available analyses: {e}")
    
    return available_analyses


"""
This function is used to get the detailed metrics for the bridge charts, and takes in the analysis type, strategy, and df_inputs. It returns the metrics dictionary with the years and the breakdown of the metrics.
"""
def calculate_all_metrics_with_breakdown(strategy="Medium", selected_analyses=None, df_inputs=None):
    """Calculate all metrics with detailed breakdown for bridge charts."""
    # Get base metrics first
    metrics = calculate_all_metrics(strategy, selected_analyses, df_inputs)
    
    # Initialize breakdown dictionary if it doesn't exist
    if 'ebitda_breakdown' not in metrics:
        metrics['ebitda_breakdown'] = {}
    
    # Extract years from the data
    years = [2022, 2023, 2024]
    
    ebitda_bridge = []
    
    # Get the original EBITDA
    if 'EBITDA' in metrics and 'original' in metrics['EBITDA']:
        initial_ebitda = metrics['EBITDA']['original'][2]  # Using the final year
    else:
        # Fallback to a default if no data
        initial_ebitda = 0
        
    # Add the starting point
    running_total = initial_ebitda
    ebitda_bridge.append({
        'label': 'Initial EBITDA',
        'value': initial_ebitda,
        'running_total': running_total
    })
    
    # Check if cloud_optimization is selected and has savings
    if 'cloud_optimization' in selected_analyses:
        # Use the 2024 cloud savings value (index 2) for consistency with other analyses
        if 'cloud_savings_original' in metrics:
            cloud_optimization = metrics['cloud_savings_original'][2]  # Use only 2024 value
        elif 'cloud_analysis' in metrics and 'total_savings' in metrics['cloud_analysis']:
            cloud_optimization = metrics['cloud_analysis']['total_savings']
        else:
            cloud_optimization = 0
            
        # Add to the breakdown for future reference
        metrics['ebitda_breakdown']['cloud_optimization'] = cloud_optimization
        
        # Add to the bridge if there's a non-zero value
        if cloud_optimization != 0:
            running_total += cloud_optimization
            ebitda_bridge.append({
                'label': 'Cloud Optimization',
                'value': cloud_optimization,
                'running_total': running_total
            })

    # Add R&D contribution if selected
    if 'rd_cost_optimization' in selected_analyses:
        if 'rd_analysis' in metrics:
            rd_contribution = metrics['rd_analysis'].get('total_savings', 0)
        else:
            rd_contribution = 0
            
        running_total += rd_contribution
        ebitda_bridge.append({
            'label': 'Rd Cost Optimization',
            'value': rd_contribution
        })
    
    # Add Support Monetization contribution if selected
    if 'support_monetization' in selected_analyses:
        support_contribution = 0 
        running_total += support_contribution
        ebitda_bridge.append({
            'label': 'Support Monetization',
            'value': support_contribution
        })
    
    # Add Other COGS contribution if selected
    if 'other_cogs_optimization' in selected_analyses:
        if 'other_cogs_optimization' in metrics['ebitda_breakdown']:
            other_cogs_contribution = metrics['ebitda_breakdown']['other_cogs_optimization']
        else:
            other_cogs_contribution = 0
            
        # Add to the breakdown for future reference
        metrics['ebitda_breakdown']['other_cogs_optimization'] = other_cogs_contribution
        
        if other_cogs_contribution != 0:
            running_total += other_cogs_contribution
            ebitda_bridge.append({
                'label': 'Other COGS Optimization',
                'value': other_cogs_contribution,
                'running_total': running_total
            })
    
    # Add Sales contribution if selected
    if 'sales_optimization' in selected_analyses and 'sales_quota_improvement' in metrics:
        sales_contribution = metrics['sales_quota_improvement'][2] * 0.6  # Using 60% modifier
        metrics['ebitda_breakdown']['sales_optimization'] = sales_contribution
        
        if sales_contribution != 0:
            running_total += sales_contribution
            ebitda_bridge.append({
                'label': 'Sales Quota Optimization',
                'value': sales_contribution,
                'running_total': running_total
            })
    
    # Add CAM contribution if selected
    if 'cam_optimization' in selected_analyses and 'cam_quota_improvement' in metrics:
        cam_contribution = metrics['cam_quota_improvement'][2] * 0.6  # Using 60% modifier
        metrics['ebitda_breakdown']['cam_optimization'] = cam_contribution
        
        if cam_contribution != 0:
            running_total += cam_contribution
            ebitda_bridge.append({
                'label': 'CAM Quota Optimization',
                'value': cam_contribution,
                'running_total': running_total
            })
    
    # Add final EBITDA
    if 'EBITDA' in metrics and 'improved' in metrics['EBITDA']:
        ebitda_bridge.append({
            'label': 'Final EBITDA',
            'value': running_total,
            'running_total': running_total
        })
    
    metrics['ebitda_bridge'] = ebitda_bridge
    
    return metrics

def calculate_cloud_savings(data, strategy, years, income_statement=None):
    """Calculate potential cloud infrastructure cost savings."""
    revenue = data['Revenue']['Plan']
    
    # Extract actual cloud costs from the input file
    found_cloud_costs = False
    actual_cloud_costs = np.zeros(len(revenue))
    
    
    if income_statement is not None:
        # Check for 'Hosting Costs' in the income statement
        if 'Hosting Costs' in income_statement['Line Item'].values:
            actual_cloud_costs = income_statement.loc[income_statement['Line Item'] == 'Hosting Costs'].iloc[:, 1:4].values.flatten()
            found_cloud_costs = True
    
    # If no actual costs found, return zero savings
    if not found_cloud_costs:
        print("No cloud costs found in income statement. Returning zeros.")
        cloud_analysis = {
            'current_pct': np.zeros(len(revenue)),
            'benchmark_pct': np.full(3, CLOUD_SAVINGS_BENCHMARKS[strategy]) * 100,
            'total_savings': 0,
            'current_costs': np.zeros(len(revenue)),
            'benchmark_costs': np.zeros(len(revenue))
        }
        return np.zeros(len(revenue)), cloud_analysis
    
    # Calculate current percentage (in decimal form)
    current_pct = actual_cloud_costs / revenue  
    print(f"Current cloud costs as percentage of revenue: {current_pct * 100}%")
    
    # Get benchmark percentage from benchmarks.py
    benchmark_pct = np.full(3, CLOUD_SAVINGS_BENCHMARKS[strategy])
    
    # Calculate costs at benchmark percentage
    benchmark_cloud_costs = benchmark_pct * revenue
    

    # Calculate savings (actual - benchmark) only if actual > benchmark (positive savings)
    potential_savings = np.maximum(0, actual_cloud_costs - benchmark_cloud_costs)

    # Return the analysis details along with savings
    cloud_analysis = {
        'current_pct': current_pct * 100,  # Convert to percentage for display
        'benchmark_pct': benchmark_pct * 100,  # Convert to percentage for display
        'total_savings': int(np.sum(potential_savings)),
        'current_costs': actual_cloud_costs,
        'benchmark_costs': benchmark_cloud_costs
    }
    
    return potential_savings, cloud_analysis

"""
This function is used to calculate the support revenue based on the premium customers, and takes in the strategy and df_inputs. It returns the support revenue for the 3 years.
"""
def calculate_support_revenue(strategy="Medium", df_inputs=None):

    # If the sheet exists, calculate based on actual data
    if 'Support Monetization' in df_inputs and not df_inputs['Support Monetization'].empty:
        df_support = df_inputs['Support Monetization']
        
        
        # Convert percentage column to numeric if it exists
        if 'Percentage of Total' in df_support.columns:
            # Try to convert percentages to numeric values
            try:
                # See if percentages are stored as strings with % sign
                if df_support['Percentage of Total'].dtype == 'object':
                    df_support['Percentage of Total'] = df_support['Percentage of Total'].str.rstrip('%').astype('float') / 100
                # else check if percentages are already numeric but formatted as percentages (e.g. 0.01 for 1%)
                elif df_support['Percentage of Total'].max() <= 1.0:
                    print("Percentages appear to be in decimal form already")
                # otherwise percentages are numeric but represented as whole numbers (e.g. 1 for 1%)
                else:
                    df_support['Percentage of Total'] = df_support['Percentage of Total'] / 100
                    print("Converted whole number percentages to decimal form")
            except Exception as e:
                print(f"Error converting percentages: {e}")
            
            # Find out how many customers exist at different thresholds of ARR concentration
            thresholds = [0.005, 0.01, 0.015, 0.02, 0.03, 0.05]
            for threshold in thresholds:
                count = df_support[df_support['Percentage of Total'] > threshold].shape[0]
        
        # Find the premium customers - those with more than 2% of total spend
        try:
            premium_threshold = 0.02  
            
            if 'Percentage of Total' in df_support.columns:
                premium_customers = df_support[df_support['Percentage of Total'] > premium_threshold].shape[0]
                
                # If we found premium customers, calculate revenue
                if premium_customers > 0:
                    # Calculate revenue: $3000 per premium customer
                    annual_revenue = premium_customers * 3000
                    
                    # Create phased implementation over three yrs
                    support_revenue = np.array([
                        annual_revenue * 0.5, 
                        annual_revenue * 0.75,  
                        annual_revenue          
                    ])
                    
                    return support_revenue
                else:
                    return np.zeros(3)
            else:
                return np.zeros(3)
        except Exception as e:
            return np.zeros(3)
    
    # If no Support Monetization sheet found, return zeros
    return np.zeros(3)

