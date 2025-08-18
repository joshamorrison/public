{{ config(
    materialized='view',
    description='Base attribution model with adstock and saturation transformations'
) }}

-- Intermediate model for attribution analysis
-- Applies adstock (carryover) and saturation (diminishing returns) transformations
WITH base_data AS (
    SELECT 
        spend_date,
        channel,
        channel_spend_usd,
        channel_volume,
        spend_year,
        spend_month,
        spend_week
    FROM {{ ref('stg_channel_performance') }}
),

-- Apply adstock transformation (carryover effects)
adstock_transformation AS (
    SELECT 
        *,
        -- Adstock rate of 0.5 (50% carryover)
        SUM(channel_spend_usd * POWER(0.5, ROW_NUMBER() OVER (PARTITION BY channel ORDER BY spend_date DESC) - 1))
            OVER (
                PARTITION BY channel 
                ORDER BY spend_date 
                ROWS BETWEEN CURRENT ROW AND 3 FOLLOWING
            ) AS channel_spend_adstocked,
            
        SUM(channel_volume * POWER(0.5, ROW_NUMBER() OVER (PARTITION BY channel ORDER BY spend_date DESC) - 1))
            OVER (
                PARTITION BY channel 
                ORDER BY spend_date 
                ROWS BETWEEN CURRENT ROW AND 3 FOLLOWING
            ) AS channel_volume_adstocked

    FROM base_data
),

-- Apply saturation transformation (diminishing returns)
saturation_transformation AS (
    SELECT 
        *,
        -- Hill saturation transformation with alpha = 0.5
        channel_spend_adstocked / (channel_spend_adstocked + 
            (MAX(channel_spend_adstocked) OVER (PARTITION BY channel) * 0.5)) AS spend_saturation_factor,
        
        channel_volume_adstocked / (channel_volume_adstocked + 
            (MAX(channel_volume_adstocked) OVER (PARTITION BY channel) * 0.5)) AS volume_saturation_factor
            
    FROM adstock_transformation
),

-- Calculate transformed metrics
final_transformations AS (
    SELECT 
        spend_date,
        channel,
        spend_year,
        spend_month,
        spend_week,
        
        -- Original metrics
        channel_spend_usd,
        channel_volume,
        
        -- Adstocked metrics
        channel_spend_adstocked,
        channel_volume_adstocked,
        
        -- Saturated metrics
        channel_spend_adstocked * spend_saturation_factor AS channel_spend_transformed,
        channel_volume_adstocked * volume_saturation_factor AS channel_volume_transformed,
        
        -- Transformation factors for analysis
        spend_saturation_factor,
        volume_saturation_factor,
        
        -- Rolling attribution windows
        SUM(channel_spend_adstocked * spend_saturation_factor) OVER (
            PARTITION BY channel 
            ORDER BY spend_date 
            ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
        ) AS channel_contribution_4wk,
        
        SUM(channel_spend_adstocked * spend_saturation_factor) OVER (
            PARTITION BY channel 
            ORDER BY spend_date 
            ROWS BETWEEN 7 PRECEDING AND CURRENT ROW
        ) AS channel_contribution_8wk

    FROM saturation_transformation
)

SELECT 
    *,
    -- Channel efficiency based on transformed metrics
    CASE 
        WHEN channel_volume_transformed > 0 
        THEN channel_spend_transformed / channel_volume_transformed 
        ELSE NULL 
    END AS transformed_efficiency_ratio,
    
    -- Contribution share
    channel_contribution_4wk / NULLIF(SUM(channel_contribution_4wk) OVER (PARTITION BY spend_date), 0) AS contribution_share_4wk

FROM final_transformations
ORDER BY spend_date, channel