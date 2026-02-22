-- ============================================================================
-- Description: This script harmonizes the Revenue and Streams datasets by
--              standardizing DSPs and Territories, aggregating the data,
--              and merging them into a single unified view.
-- ============================================================================

-- 1. Standardize Revenue Data
-- Map granular DSPs to the 5 major DSPs
WITH revenue_standardized AS (
    SELECT
        month,
        business_unit,
        territory_name,
        CASE
            WHEN dsp IN (
                'YouTube Premium Individual monthly', 'YouTube - Ads', 'YouTube - Premium',
                'YouTube Ad Revenue', 'YouTube Premium Family monthly', 'YouTube - Audio Tier',
                'YouTube', 'YouTube Premium Student monthly', 'YouTube Music Family monthly',
                'YouTube Music Student monthly', 'YouTube Music', 'YouTube Music Individual monthly',
                'YouTube Premium Lite Individual monthly', 'Youtube Licensing', 'YouTube Premium',
                'YouTube - Other'
            ) THEN 'YouTube'
            WHEN dsp IN ('Tiktok') THEN 'TikTok'
            WHEN dsp IN ('Spotify') THEN 'Spotify'
            WHEN dsp IN ('Amazon', 'Amazon Prime', 'Amazon Unlimited', 'Amazon Ad Supported', 'Amazon Cloud') THEN 'Amazon'
            WHEN dsp IN ('Apple/iTunes', 'Apple Music Dj Mixes', 'Apple Music', 'Apple Inc.') THEN 'Apple'
            ELSE 'Other'
        END AS dsp,
        total_gross_amount
    FROM revenue
),

-- 2. Aggregate Revenue Data
-- Group by month, BU, territory, and standardized DSP
revenue_aggregated AS (
    SELECT
        month,
        business_unit,
        territory_name,
        dsp,
        SUM(total_gross_amount) AS total_gross_amount
    FROM revenue_standardized
    GROUP BY
        month,
        business_unit,
        territory_name,
        dsp
),

-- 3. Standardize Streams Data
streams_standardized AS (
    SELECT
        month,
        business_unit,
        CASE
            WHEN country IN (
                'Brazil', 'Canada', 'United States of America', 'Germany',
                'United Kingdom', 'France', 'India', 'Australia', 'Mexico'
            ) THEN country
            ELSE 'All Other Locations'
        END AS territory_name,
        dsp,
        total_streams
    FROM streams
),

-- 4. Aggregate Streams Data
-- Group by month, BU, standardized territory, and DSP
streams_aggregated AS (
    SELECT
        month,
        business_unit,
        territory_name,
        dsp,
        SUM(total_streams) AS total_streams
    FROM streams_standardized
    GROUP BY
        month,
        business_unit,
        territory_name,
        dsp
),

-- 5. Merge Datasets
merged_data AS (
    SELECT
        COALESCE(s.month, r.month) AS month,
        COALESCE(s.business_unit, r.business_unit) AS business_unit,
        COALESCE(s.territory_name, r.territory_name) AS territory_name,
        COALESCE(s.dsp, r.dsp) AS dsp,
        s.total_streams,
        r.total_gross_amount
    FROM streams_aggregated s
    FULL OUTER JOIN revenue_aggregated r
        ON s.month = r.month
        AND s.business_unit = r.business_unit
        AND s.territory_name = r.territory_name
        AND s.dsp = r.dsp
)

-- 6. Final Output with Indicators
SELECT
    month,
    business_unit,
    territory_name,
    dsp,
    total_streams,
    total_gross_amount,
    -- Indicator for missing revenue
    CASE WHEN total_gross_amount IS NULL THEN 1 ELSE 0 END AS is_na,
    -- Indicator for zero revenue
    CASE WHEN total_gross_amount = 0 THEN 1 ELSE 0 END AS is_zero
FROM merged_data;

-- ============================================================================
-- Data Quality Checks
-- ============================================================================


-- Check 1: Ensure no missing values in key dimensions
SELECT 'Missing Dimensions Check' AS check_name, COUNT(*) AS failed_rows
FROM final_table 
WHERE month IS NULL OR business_unit IS NULL OR territory_name IS NULL OR dsp IS NULL;

-- Check 2: Ensure total revenue matches the source
SELECT 
    'Revenue Total Check' AS check_name,
    (SELECT SUM(total_gross_amount) FROM revenue) AS source_revenue,
    (SELECT SUM(total_gross_amount) FROM final_table) AS final_revenue,
    CASE WHEN (SELECT SUM(total_gross_amount) FROM revenue) = (SELECT SUM(total_gross_amount) FROM final_table) THEN 'PASS' ELSE 'FAIL' END AS status;

-- Check 3: Ensure total streams match the source
SELECT 
    'Streams Total Check' AS check_name,
    (SELECT SUM(total_streams) FROM streams) AS source_streams,
    (SELECT SUM(total_streams) FROM final_table) AS final_streams,
    CASE WHEN (SELECT SUM(total_streams) FROM streams) = (SELECT SUM(total_streams) FROM final_table) THEN 'PASS' ELSE 'FAIL' END AS status;
