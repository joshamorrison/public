{{ config(
    materialized='view',
    description='Multiple attribution methods comparison (first-touch, last-touch, data-driven)'
) }}

-- Intermediate model comparing different attribution methodologies
WITH base_attribution AS (
    SELECT 
        spend_date,
        channel,
        channel_spend_transformed,
        channel_volume_transformed,
        contribution_share_4wk,
        spend_year,
        spend_month,
        spend_week
    FROM {{ ref('int_attribution_base') }}
),

-- Get total metrics by date for attribution calculations
date_totals AS (
    SELECT 
        s.spend_date,
        s.total_conversions,
        s.total_revenue_usd,
        s.total_media_spend_usd
    FROM {{ ref('stg_media_spend') }} s
),

-- Join attribution base with performance totals
attribution_joined AS (
    SELECT 
        b.*,
        t.total_conversions,
        t.total_revenue_usd,
        t.total_media_spend_usd
    FROM base_attribution b
    LEFT JOIN date_totals t ON b.spend_date = t.spend_date
),

-- Calculate different attribution methods
attribution_methods AS (
    SELECT 
        spend_date,
        channel,
        spend_year,
        spend_month,
        spend_week,
        channel_spend_transformed,
        total_conversions,
        total_revenue_usd,
        
        -- Linear Attribution (equal weight)
        total_conversions / 5.0 AS linear_attributed_conversions,
        total_revenue_usd / 5.0 AS linear_attributed_revenue,
        
        -- Spend-Based Attribution (proportional to spend)
        total_conversions * (channel_spend_transformed / NULLIF(total_media_spend_usd, 0)) AS spend_based_conversions,
        total_revenue_usd * (channel_spend_transformed / NULLIF(total_media_spend_usd, 0)) AS spend_based_revenue,
        
        -- Data-Driven Attribution (based on transformed contribution)
        total_conversions * COALESCE(contribution_share_4wk, 0) AS data_driven_conversions,
        total_revenue_usd * COALESCE(contribution_share_4wk, 0) AS data_driven_revenue,
        
        -- First-Touch Attribution (highest spend gets 60%, others split 40%)
        CASE 
            WHEN channel_spend_transformed = MAX(channel_spend_transformed) OVER (PARTITION BY spend_date)
            THEN total_conversions * 0.6
            ELSE total_conversions * 0.4 / 4.0
        END AS first_touch_conversions,
        
        CASE 
            WHEN channel_spend_transformed = MAX(channel_spend_transformed) OVER (PARTITION BY spend_date)
            THEN total_revenue_usd * 0.6
            ELSE total_revenue_usd * 0.4 / 4.0
        END AS first_touch_revenue,
        
        -- Time-Decay Attribution (more recent periods get higher weight)
        total_conversions * (
            POWER(0.9, EXTRACT(dayofyear FROM CURRENT_DATE) - EXTRACT(dayofyear FROM spend_date)) * 
            (channel_spend_transformed / NULLIF(total_media_spend_usd, 0))
        ) AS time_decay_conversions,
        
        total_revenue_usd * (
            POWER(0.9, EXTRACT(dayofyear FROM CURRENT_DATE) - EXTRACT(dayofyear FROM spend_date)) * 
            (channel_spend_transformed / NULLIF(total_media_spend_usd, 0))
        ) AS time_decay_revenue

    FROM attribution_joined
    WHERE total_conversions > 0 AND total_revenue_usd > 0
),

-- Calculate ROI for each attribution method
attribution_roi AS (
    SELECT 
        *,
        -- ROI calculations for each method
        CASE WHEN channel_spend_transformed > 0 THEN linear_attributed_revenue / channel_spend_transformed ELSE NULL END AS linear_roi,
        CASE WHEN channel_spend_transformed > 0 THEN spend_based_revenue / channel_spend_transformed ELSE NULL END AS spend_based_roi,
        CASE WHEN channel_spend_transformed > 0 THEN data_driven_revenue / channel_spend_transformed ELSE NULL END AS data_driven_roi,
        CASE WHEN channel_spend_transformed > 0 THEN first_touch_revenue / channel_spend_transformed ELSE NULL END AS first_touch_roi,
        CASE WHEN channel_spend_transformed > 0 THEN time_decay_revenue / channel_spend_transformed ELSE NULL END AS time_decay_roi,
        
        -- CAC calculations for each method
        CASE WHEN linear_attributed_conversions > 0 THEN channel_spend_transformed / linear_attributed_conversions ELSE NULL END AS linear_cac,
        CASE WHEN spend_based_conversions > 0 THEN channel_spend_transformed / spend_based_conversions ELSE NULL END AS spend_based_cac,
        CASE WHEN data_driven_conversions > 0 THEN channel_spend_transformed / data_driven_conversions ELSE NULL END AS data_driven_cac,
        CASE WHEN first_touch_conversions > 0 THEN channel_spend_transformed / first_touch_conversions ELSE NULL END AS first_touch_cac,
        CASE WHEN time_decay_conversions > 0 THEN channel_spend_transformed / time_decay_conversions ELSE NULL END AS time_decay_cac

    FROM attribution_methods
)

SELECT *
FROM attribution_roi
ORDER BY spend_date, channel