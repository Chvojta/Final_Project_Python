CLOUD_SAVINGS_BENCHMARKS = {
    "High": 0.075,   # High = aggressive = 3% of revenue (significantly more aggressive)
    "Medium": 0.1,  # Medium = 10% of revenue 
    "Low": 0.15     # Low = 15% of revenue
}

OTHER_COGS_BENCHMARKS = {
    "High": 0.07,   # High = aggressive = 2.5% of revenue (more aggressive)
    "Medium": 0.085, # Medium = 7% of revenue
    "Low": 0.1       # Low = 10% of revenue
}

salary_benchmarks = {
    "Software Engineer I": {
        "low": 250000,     # Higher salary benchmark = lower savings (conservative approach)
        "medium": 125000,  # Medium salary benchmark
        "high": 80000      # Lower salary benchmark = higher savings (aggressive approach)
    },
    "Software Engineer II": {
        "low": 312500,
        "medium": 156250,
        "high": 100000
    },
    "Senior Software Engineer": {
        "low": 500000,
        "medium": 250000,
        "high": 160000
    },
    "Junior DevOps Engineer": {
        "low": 150000,     # Fixed - swapped high/low for consistency
        "medium": 120000,
        "high": 80000      # Fixed - lowest salary = highest savings
    },
    "Senior DevOps Engineer": {
        "low": 350000,     # Fixed - swapped high/low for consistency
        "medium": 200000,
        "high": 120000     # Fixed - lowest salary = highest savings
    },
    "QA Engineer": {
        "low": 150000,     # Fixed - swapped high/low for consistency
        "medium": 100000,
        "high": 65000
    },
    "Junior Back-end Engineer": {
        "low": 250000,     # Fixed - swapped high/low for consistency
        "medium": 125000,
        "high": 80000
    },
    "Senior Back-end Engineer": {
        "low": 500000,     # Fixed - swapped high/low for consistency
        "medium": 250000,
        "high": 160000
    },
    "Junior Cloud Architect": {
        "low": 200000,     # Fixed - swapped high/low for consistency
        "medium": 120000,
        "high": 80000
    },
    "Product Manager I": {
        "low": 200000,     # Fixed - swapped high/low for consistency
        "medium": 120000,
        "high": 100000
    },
    "Senior Cloud Architect": {
        "low": 350000,     # Fixed - swapped high/low for consistency
        "medium": 200000,
        "high": 120000
    }
}
