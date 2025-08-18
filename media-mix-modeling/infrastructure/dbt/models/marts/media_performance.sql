{{ config(
    materialized='table',
    description='Final media performance metrics with comprehensive attribution analysis'
) }}

-- Mart model for media performance reporting
-- Provides executive-ready metrics with multiple attribution methods
WITH attribution_summary AS (
    SELECT 
        spend_date,
        channel,
        spend_year,
        spend_month,
        spend_week,
        channel_spend_transformed,
        
        -- Attribution methods
        linear_attributed_conversions,
        linear_attributed_revenue,
        spend_based_conversions,
        spend_based_revenue,
        data_driven_conversions,
        data_driven_revenue,
        
        -- ROI by method
        linear_roi,
        spend_based_roi,
        data_driven_roi,
        
        -- CAC by method
        linear_cac,
        spend_based_cac,
        data_driven_cac

    FROM {{ ref('int_attribution_methods') }}
),

-- Calculate aggregated metrics
channel_metrics AS (
    SELECT 
        channel,
        spend_year,
        spend_month,
        
        -- Spend metrics
        SUM(channel_spend_transformed) AS total_spend,
        AVG(channel_spend_transformed) AS avg_weekly_spend,
        COUNT(*) AS weeks_active,
        
        -- Attribution totals
        SUM(linear_attributed_conversions) AS total_linear_conversions,
        SUM(linear_attributed_revenue) AS total_linear_revenue,
        SUM(spend_based_conversions) AS total_spend_based_conversions,
        SUM(spend_based_revenue) AS total_spend_based_revenue,
        SUM(data_driven_conversions) AS total_data_driven_conversions,
        SUM(data_driven_revenue) AS total_data_driven_revenue,
        
        -- Average ROI
        AVG(linear_roi) AS avg_linear_roi,
        AVG(spend_based_roi) AS avg_spend_based_roi,
        AVG(data_driven_roi) AS avg_data_driven_roi,
        
        -- Average CAC
        AVG(linear_cac) AS avg_linear_cac,
        AVG(spend_based_cac) AS avg_spend_based_cac,
        AVG(data_driven_cac) AS avg_data_driven_cac

    FROM attribution_summary
    GROUP BY channel, spend_year, spend_month
),

-- Portfolio-level metrics
portfolio_metrics AS (
    SELECT 
        spend_year,
        spend_month,
        SUM(total_spend) AS portfolio_total_spend,
        SUM(total_data_driven_revenue) AS portfolio_total_revenue,
        SUM(total_data_driven_conversions) AS portfolio_total_conversions,
        
        -- Portfolio ROI (using data-driven attribution as default)
        CASE 
            WHEN SUM(total_spend) > 0 
            THEN SUM(total_data_driven_revenue) / SUM(total_spend) 
            ELSE NULL 
        END AS portfolio_roi,
        
        -- Portfolio CAC
        CASE 
            WHEN SUM(total_data_driven_conversions) > 0 
            THEN SUM(total_spend) / SUM(total_data_driven_conversions) 
            ELSE NULL 
        END AS portfolio_cac

    FROM channel_metrics
    GROUP BY spend_year, spend_month
),

-- Channel performance with benchmarks
channel_performance AS (
    SELECT 
        c.*,
        p.portfolio_total_spend,
        p.portfolio_roi,
        p.portfolio_cac,
        
        -- Channel share metrics
        c.total_spend / p.portfolio_total_spend AS channel_spend_share,
        c.total_data_driven_revenue / p.portfolio_total_revenue AS channel_revenue_share,
        
        -- Performance vs portfolio
        c.avg_data_driven_roi / p.portfolio_roi AS roi_vs_portfolio,
        c.avg_data_driven_cac / p.portfolio_cac AS cac_vs_portfolio,
        
        -- Efficiency rankings
        RANK() OVER (
            PARTITION BY c.spend_year, c.spend_month 
            ORDER BY c.avg_data_driven_roi DESC
        ) AS roi_rank,
        
        RANK() OVER (
            PARTITION BY c.spend_year, c.spend_month 
            ORDER BY c.avg_data_driven_cac ASC
        ) AS cac_rank,
        
        -- Attribution method variance (to assess reliability)
        ABS(c.avg_linear_roi - c.avg_data_driven_roi) AS linear_vs_dd_roi_variance,
        ABS(c.avg_spend_based_roi - c.avg_data_driven_roi) AS spend_vs_dd_roi_variance

    FROM channel_metrics c
    LEFT JOIN portfolio_metrics p 
        ON c.spend_year = p.spend_year 
        AND c.spend_month = p.spend_month
),

-- Final enrichment with recommendations
final_performance AS (
    SELECT 
        *,
        -- Performance categories
        CASE 
            WHEN roi_rank <= 2 AND cac_rank <= 2 THEN 'Top Performer'
            WHEN roi_rank <= 3 OR cac_rank <= 3 THEN 'Strong Performer' 
            WHEN roi_vs_portfolio > 1.1 THEN 'Above Average'
            WHEN roi_vs_portfolio BETWEEN 0.9 AND 1.1 THEN 'Average'
            ELSE 'Below Average'
        END AS performance_category,
        
        -- Budget recommendations
        CASE 
            WHEN roi_vs_portfolio > 1.2 AND cac_vs_portfolio < 0.8 THEN 'Increase Budget'
            WHEN roi_vs_portfolio > 1.0 AND channel_spend_share < 0.15 THEN 'Scale Gradually'
            WHEN roi_vs_portfolio BETWEEN 0.9 AND 1.1 THEN 'Maintain Budget'
            WHEN roi_vs_portfolio < 0.8 OR cac_vs_portfolio > 1.3 THEN 'Reduce Budget'
            ELSE 'Monitor Closely'
        END AS budget_recommendation,
        
        -- Attribution reliability
        CASE 
            WHEN linear_vs_dd_roi_variance < 0.2 AND spend_vs_dd_roi_variance < 0.2 THEN 'High Confidence'
            WHEN linear_vs_dd_roi_variance < 0.5 AND spend_vs_dd_roi_variance < 0.5 THEN 'Medium Confidence'
            ELSE 'Low Confidence'
        END AS attribution_confidence

    FROM channel_performance
)

SELECT 
    channel,
    spend_year,
    spend_month,
    total_spend,
    avg_weekly_spend,
    weeks_active,
    
    -- Primary metrics (data-driven attribution)
    total_data_driven_conversions AS total_conversions,
    total_data_driven_revenue AS total_revenue,
    avg_data_driven_roi AS avg_roi,
    avg_data_driven_cac AS avg_cac,
    
    -- Portfolio context
    channel_spend_share,
    channel_revenue_share,
    portfolio_roi,
    portfolio_cac,
    
    -- Performance indicators
    performance_category,
    budget_recommendation,
    attribution_confidence,
    roi_rank,
    cac_rank,
    roi_vs_portfolio,
    cac_vs_portfolio,
    
    -- Alternative attribution methods for comparison
    avg_linear_roi,
    avg_spend_based_roi,
    linear_vs_dd_roi_variance,
    spend_vs_dd_roi_variance

FROM final_performance
ORDER BY spend_year DESC, spend_month DESC, roi_rank ASC