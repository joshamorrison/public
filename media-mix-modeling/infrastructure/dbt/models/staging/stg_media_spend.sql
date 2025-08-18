{{ config(
    materialized='view',
    description='Cleaned and standardized media spend data from multiple sources'
) }}

-- Staging model for media spend data
-- Standardizes column naming, data types, and applies basic cleaning
WITH media_spend_raw AS (
    SELECT
        date,
        tv_spend,
        digital_spend,
        radio_spend,
        print_spend,
        social_spend,
        tv_impressions,
        digital_clicks,
        radio_reach,
        print_circulation,
        social_engagement,
        conversions,
        revenue
    FROM {{ source('raw', 'marketing_data') }}
),

standardized AS (
    SELECT
        date::date AS spend_date,
        tv_spend::decimal(12,2) AS tv_spend_usd,
        digital_spend::decimal(12,2) AS digital_spend_usd,
        radio_spend::decimal(12,2) AS radio_spend_usd,
        print_spend::decimal(12,2) AS print_spend_usd,
        social_spend::decimal(12,2) AS social_spend_usd,
        
        -- Media impressions/reach metrics
        tv_impressions::integer AS tv_impressions,
        digital_clicks::integer AS digital_clicks,
        radio_reach::integer AS radio_reach,
        print_circulation::integer AS print_circulation,
        social_engagement::integer AS social_engagement,
        
        -- Performance metrics
        conversions::integer AS total_conversions,
        revenue::decimal(12,2) AS total_revenue_usd,
        
        -- Calculated fields
        (tv_spend + digital_spend + radio_spend + print_spend + social_spend)::decimal(12,2) AS total_media_spend_usd,
        
        -- Week over week calculations
        LAG(tv_spend + digital_spend + radio_spend + print_spend + social_spend) 
            OVER (ORDER BY date) AS prev_week_total_spend,
        
        -- Date attributes
        EXTRACT(year FROM date) AS spend_year,
        EXTRACT(month FROM date) AS spend_month,
        EXTRACT(week FROM date) AS spend_week,
        EXTRACT(dayofweek FROM date) AS spend_day_of_week

    FROM media_spend_raw
    WHERE date IS NOT NULL
      AND (tv_spend + digital_spend + radio_spend + print_spend + social_spend) > 0
)

SELECT 
    *,
    -- Week over week growth
    CASE 
        WHEN prev_week_total_spend > 0 
        THEN (total_media_spend_usd - prev_week_total_spend) / prev_week_total_spend * 100
        ELSE NULL 
    END AS wow_spend_growth_pct,
    
    -- Efficiency metrics
    CASE 
        WHEN total_conversions > 0 
        THEN total_media_spend_usd / total_conversions 
        ELSE NULL 
    END AS cost_per_conversion,
    
    CASE 
        WHEN total_revenue_usd > 0 
        THEN total_media_spend_usd / total_revenue_usd 
        ELSE NULL 
    END AS spend_to_revenue_ratio

FROM standardized
ORDER BY spend_date