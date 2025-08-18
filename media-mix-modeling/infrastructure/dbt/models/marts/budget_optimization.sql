{{ config(
    materialized='table',
    description='Budget optimization recommendations based on marginal ROI analysis'
) }}

-- Mart model for budget optimization recommendations
-- Provides optimal budget allocation based on performance curves
WITH current_performance AS (
    SELECT 
        channel,
        AVG(avg_roi) AS current_roi,
        AVG(avg_cac) AS current_cac,
        AVG(total_spend) AS current_monthly_spend,
        AVG(channel_spend_share) AS current_spend_share,
        AVG(total_conversions) AS current_monthly_conversions,
        AVG(total_revenue) AS current_monthly_revenue
    FROM {{ ref('media_performance') }}
    WHERE spend_year = EXTRACT(year FROM CURRENT_DATE)
      AND spend_month >= EXTRACT(month FROM CURRENT_DATE) - 3  -- Last 3 months
    GROUP BY channel
),

-- Marginal efficiency analysis
marginal_analysis AS (
    SELECT 
        channel,
        current_roi,
        current_cac,
        current_monthly_spend,
        current_spend_share,
        current_monthly_conversions,
        current_monthly_revenue,
        
        -- Estimated marginal ROI (assuming diminishing returns)
        current_roi * POWER(0.95, current_spend_share * 10) AS marginal_roi,
        
        -- Efficiency vs spend relationship
        current_roi / current_spend_share AS roi_per_share_point,
        
        -- Elasticity estimate (response to 10% budget change)
        CASE 
            WHEN current_spend_share > 0.4 THEN 0.3  -- High saturation
            WHEN current_spend_share > 0.2 THEN 0.6  -- Medium saturation
            ELSE 0.9  -- Low saturation
        END AS spend_elasticity

    FROM current_performance
),

-- Portfolio totals for reallocation
portfolio_totals AS (
    SELECT 
        SUM(current_monthly_spend) AS total_portfolio_spend,
        SUM(current_monthly_revenue) AS total_portfolio_revenue,
        SUM(current_monthly_conversions) AS total_portfolio_conversions,
        AVG(current_roi) AS avg_portfolio_roi
    FROM marginal_analysis
),

-- Optimization scenarios
optimization_scenarios AS (
    SELECT 
        m.*,
        p.total_portfolio_spend,
        p.avg_portfolio_roi,
        
        -- Scenario 1: Maximize ROI
        CASE 
            WHEN marginal_roi > p.avg_portfolio_roi * 1.1 THEN current_monthly_spend * 1.2
            WHEN marginal_roi > p.avg_portfolio_roi THEN current_monthly_spend * 1.1
            WHEN marginal_roi < p.avg_portfolio_roi * 0.8 THEN current_monthly_spend * 0.8
            ELSE current_monthly_spend
        END AS roi_optimized_spend,
        
        -- Scenario 2: Diversified approach (reduce concentration risk)
        CASE 
            WHEN current_spend_share > 0.35 THEN current_monthly_spend * 0.9
            WHEN current_spend_share < 0.15 AND marginal_roi > p.avg_portfolio_roi THEN current_monthly_spend * 1.15
            ELSE current_monthly_spend
        END AS diversified_spend,
        
        -- Scenario 3: Growth-focused (prioritize volume)
        CASE 
            WHEN spend_elasticity > 0.7 AND marginal_roi > p.avg_portfolio_roi * 0.9 THEN current_monthly_spend * 1.25
            WHEN spend_elasticity > 0.5 THEN current_monthly_spend * 1.1
            ELSE current_monthly_spend * 0.95
        END AS growth_focused_spend

    FROM marginal_analysis m
    CROSS JOIN portfolio_totals p
),

-- Normalize optimized budgets to maintain total spend
normalized_budgets AS (
    SELECT 
        *,
        -- ROI scenario normalization
        roi_optimized_spend * (total_portfolio_spend / SUM(roi_optimized_spend) OVER ()) AS roi_normalized_spend,
        
        -- Diversified scenario normalization  
        diversified_spend * (total_portfolio_spend / SUM(diversified_spend) OVER ()) AS diversified_normalized_spend,
        
        -- Growth scenario normalization
        growth_focused_spend * (total_portfolio_spend / SUM(growth_focused_spend) OVER ()) AS growth_normalized_spend

    FROM optimization_scenarios
),

-- Calculate projected impact
impact_projections AS (
    SELECT 
        channel,
        current_monthly_spend,
        current_monthly_revenue,
        current_monthly_conversions,
        current_roi,
        current_cac,
        marginal_roi,
        spend_elasticity,
        
        -- ROI optimization scenario
        roi_normalized_spend,
        roi_normalized_spend - current_monthly_spend AS roi_budget_change,
        (roi_normalized_spend - current_monthly_spend) / current_monthly_spend * 100 AS roi_budget_change_pct,
        
        -- Projected performance for ROI scenario
        current_monthly_revenue * (1 + ((roi_normalized_spend - current_monthly_spend) / current_monthly_spend) * spend_elasticity) AS roi_projected_revenue,
        current_monthly_conversions * (1 + ((roi_normalized_spend - current_monthly_spend) / current_monthly_spend) * spend_elasticity) AS roi_projected_conversions,
        
        -- Diversified scenario
        diversified_normalized_spend,
        diversified_normalized_spend - current_monthly_spend AS div_budget_change,
        (diversified_normalized_spend - current_monthly_spend) / current_monthly_spend * 100 AS div_budget_change_pct,
        
        -- Growth scenario
        growth_normalized_spend,
        growth_normalized_spend - current_monthly_spend AS growth_budget_change,
        (growth_normalized_spend - current_monthly_spend) / current_monthly_spend * 100 AS growth_budget_change_pct

    FROM normalized_budgets
),

-- Final recommendations
final_recommendations AS (
    SELECT 
        *,
        -- Projected ROI for ROI scenario
        CASE 
            WHEN roi_normalized_spend > 0 
            THEN roi_projected_revenue / roi_normalized_spend 
            ELSE NULL 
        END AS roi_scenario_projected_roi,
        
        -- Revenue lift for ROI scenario
        roi_projected_revenue - current_monthly_revenue AS roi_scenario_revenue_lift,
        
        -- Recommended scenario based on current performance
        CASE 
            WHEN current_roi > avg_portfolio_roi * 1.2 AND spend_elasticity > 0.6 THEN 'Growth Focused'
            WHEN current_spend_share > 0.4 THEN 'Diversified'
            WHEN marginal_roi > avg_portfolio_roi * 1.1 THEN 'ROI Optimized'
            ELSE 'Maintain Current'
        END AS recommended_scenario,
        
        -- Confidence level
        CASE 
            WHEN spend_elasticity > 0.7 THEN 'High'
            WHEN spend_elasticity > 0.4 THEN 'Medium'
            ELSE 'Low'
        END AS recommendation_confidence

    FROM impact_projections
    CROSS JOIN (SELECT AVG(current_roi) AS avg_portfolio_roi FROM current_performance) p
)

SELECT 
    channel,
    
    -- Current state
    current_monthly_spend,
    current_monthly_revenue,
    current_monthly_conversions,
    current_roi,
    current_cac,
    
    -- Optimization insights
    marginal_roi,
    spend_elasticity,
    recommended_scenario,
    recommendation_confidence,
    
    -- ROI optimization scenario
    roi_normalized_spend AS roi_optimized_budget,
    roi_budget_change,
    roi_budget_change_pct,
    roi_scenario_projected_roi,
    roi_scenario_revenue_lift,
    
    -- Alternative scenarios
    diversified_normalized_spend AS diversified_budget,
    div_budget_change_pct AS diversified_change_pct,
    growth_normalized_spend AS growth_budget,
    growth_budget_change_pct AS growth_change_pct,
    
    -- Decision support
    CASE 
        WHEN ABS(roi_budget_change_pct) < 5 THEN 'Minor Adjustment'
        WHEN ABS(roi_budget_change_pct) < 15 THEN 'Moderate Adjustment'
        ELSE 'Major Reallocation'
    END AS change_magnitude,
    
    CASE 
        WHEN roi_scenario_revenue_lift > current_monthly_revenue * 0.1 THEN 'High Impact'
        WHEN roi_scenario_revenue_lift > current_monthly_revenue * 0.05 THEN 'Medium Impact'
        ELSE 'Low Impact'
    END AS expected_impact

FROM final_recommendations
ORDER BY roi_scenario_revenue_lift DESC