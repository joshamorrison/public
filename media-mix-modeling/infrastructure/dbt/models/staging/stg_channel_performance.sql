{{ config(
    materialized='view',
    description='Channel-level performance metrics with standardized KPIs'
) }}

-- Staging model for channel performance data
-- Unpivots spend data and calculates channel-level metrics
WITH spend_unpivoted AS (
    SELECT 
        spend_date,
        spend_year,
        spend_month,
        spend_week,
        total_conversions,
        total_revenue_usd,
        
        -- Unpivot channel spend
        'tv' as channel,
        tv_spend_usd as channel_spend_usd,
        tv_impressions as channel_volume
    FROM {{ ref('stg_media_spend') }}
    
    UNION ALL
    
    SELECT 
        spend_date,
        spend_year,
        spend_month,
        spend_week,
        total_conversions,
        total_revenue_usd,
        'digital' as channel,
        digital_spend_usd as channel_spend_usd,
        digital_clicks as channel_volume
    FROM {{ ref('stg_media_spend') }}
    
    UNION ALL
    
    SELECT 
        spend_date,
        spend_year,
        spend_month,
        spend_week,
        total_conversions,
        total_revenue_usd,
        'radio' as channel,
        radio_spend_usd as channel_spend_usd,
        radio_reach as channel_volume
    FROM {{ ref('stg_media_spend') }}
    
    UNION ALL
    
    SELECT 
        spend_date,
        spend_year,
        spend_month,
        spend_week,
        total_conversions,
        total_revenue_usd,
        'print' as channel,
        print_spend_usd as channel_spend_usd,
        print_circulation as channel_volume
    FROM {{ ref('stg_media_spend') }}
    
    UNION ALL
    
    SELECT 
        spend_date,
        spend_year,
        spend_month,
        spend_week,
        total_conversions,
        total_revenue_usd,
        'social' as channel,
        social_spend_usd as channel_spend_usd,
        social_engagement as channel_volume
    FROM {{ ref('stg_media_spend') }}
),

channel_metrics AS (
    SELECT
        *,
        -- Channel efficiency metrics
        CASE 
            WHEN channel_volume > 0 
            THEN channel_spend_usd / channel_volume * 1000 
            ELSE NULL 
        END AS cost_per_thousand_volume,
        
        -- Moving averages (4-week)
        AVG(channel_spend_usd) OVER (
            PARTITION BY channel 
            ORDER BY spend_date 
            ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
        ) AS channel_spend_4wk_avg,
        
        AVG(channel_volume) OVER (
            PARTITION BY channel 
            ORDER BY spend_date 
            ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
        ) AS channel_volume_4wk_avg

    FROM spend_unpivoted
    WHERE channel_spend_usd > 0
)

SELECT 
    spend_date,
    channel,
    spend_year,
    spend_month,
    spend_week,
    channel_spend_usd,
    channel_volume,
    cost_per_thousand_volume,
    channel_spend_4wk_avg,
    channel_volume_4wk_avg,
    
    -- Share calculations
    channel_spend_usd / SUM(channel_spend_usd) OVER (PARTITION BY spend_date) AS channel_spend_share,
    
    -- Performance vs average
    CASE 
        WHEN channel_spend_4wk_avg > 0 
        THEN (channel_spend_usd - channel_spend_4wk_avg) / channel_spend_4wk_avg * 100
        ELSE NULL 
    END AS spend_vs_4wk_avg_pct

FROM channel_metrics
ORDER BY spend_date, channel