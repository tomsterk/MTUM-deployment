WITH
  customers_campaign AS (
	/* Customers that received incentive or control group */
    SELECT
      file_name AS churn_group,
      user_id AS customer_nk
    FROM
      ext_uploads.file_user_ids_uplift_20260212125800 ec
  ),
  covariates_churn AS (
    SELECT
      /* Customer variables */
      dc.customer_nk,
      dc.has_rfl,
      dc.gender,
      dc.country_sk,
      /* Lifetime transactional variables */
      SUM(fbi.price_total_excl_vat) AS sales_ttl,
      SUM(fbi.quantity) AS volume_ttl,
      COUNT(DISTINCT fbi.basket_sk) AS total_transactions,
      MIN(fbi.date_trading_nk) AS first_transaction_date,
      MAX(fbi.date_trading_nk) AS last_order_date,
      /* Lifetime channel variables */
      SUM(
        CASE
          WHEN sales_channel_l1_name LIKE 'Online' THEN fbi.price_total_excl_vat
          ELSE 0
        END
      ) AS online_sales,
      SUM(
        CASE
          WHEN sales_channel_l1_name NOT LIKE 'Online' THEN fbi.price_total_excl_vat
          ELSE 0
        END
      ) AS retail_sales,
      /* Lifetime category variables */
      SUM(
        CASE
          WHEN category_fpna_l1_name = 'Food' THEN price_total_excl_vat
          ELSE 0
        END
      ) AS food_total,
      SUM(
        CASE
          WHEN category_fpna_l1_name = 'VHMS' THEN price_total_excl_vat
          ELSE 0
        END
      ) AS vhms_total,
      SUM(
        CASE
          WHEN category_fpna_l1_name = 'Active Nutrition' THEN price_total_excl_vat
          ELSE 0
        END
      ) AS sports_total,
      SUM(
        CASE
          WHEN category_fpna_l1_name = 'Beauty' THEN price_total_excl_vat
          ELSE 0
        END
      ) AS beauty_total,
      /* Last year variables (52w) within the pre-churn window. */
      COUNT(
        DISTINCT CASE
          WHEN fbi.date_trading_nk >= add_months('2026-02-12', -24) THEN fbi.basket_sk
        END
      ) AS num_orders_52wk,
      SUM(
        CASE
          WHEN fbi.date_trading_nk >= add_months('2026-02-12', -24) THEN fbi.price_total_excl_vat
          ELSE 0
        END
      ) AS sales_52wk,
      SUM(
        CASE
          WHEN fbi.date_trading_nk >= add_months('2026-02-12', -24) THEN fbi.quantity
          ELSE 0
        END
      ) AS volume_52wk,
      SUM(
        CASE
          WHEN fbi.date_trading_nk >= add_months('2026-02-12', -24)
          AND sales_channel_l1_name LIKE 'Online' THEN fbi.price_total_excl_vat
          ELSE 0
        END
      ) AS online_sales_52w,
      SUM(
        CASE
          WHEN fbi.date_trading_nk >= add_months('2026-02-12', -24)
          AND sales_channel_l1_name NOT LIKE 'Online' THEN fbi.price_total_excl_vat
          ELSE 0
        END
      ) AS retail_sales_52w,
      /* Last year to two-year variables (53w_104w) */
      COUNT(
        DISTINCT CASE
          WHEN fbi.date_trading_nk >= add_months('2026-02-12', -36)
          AND fbi.date_trading_nk < add_months('2026-02-12', -24) THEN fbi.basket_sk
        END
      ) AS num_orders_53w_104w,
      SUM(
        CASE
          WHEN fbi.date_trading_nk >= add_months('2026-02-12', -36)
          AND fbi.date_trading_nk < add_months('2026-02-12', -24) THEN fbi.price_total_excl_vat
          ELSE 0
        END
      ) AS sales_53w_104w,
      SUM(
        CASE
          WHEN fbi.date_trading_nk >= add_months('2026-02-12', -36)
          AND fbi.date_trading_nk < add_months('2026-02-12', -24) THEN fbi.quantity
          ELSE 0
        END
      ) AS volume_53w_104w,
      SUM(
        CASE
          WHEN fbi.date_trading_nk >= add_months('2026-02-12', -36)
          AND fbi.date_trading_nk < add_months('2026-02-12', -24)
          AND sales_channel_l1_name LIKE 'Online' THEN fbi.price_total_excl_vat
          ELSE 0
        END
      ) AS online_sales_53w_104w,
      SUM(
        CASE
          WHEN fbi.date_trading_nk >= add_months('2026-02-12', -36)
          AND fbi.date_trading_nk < add_months('2026-02-12', -24)
          AND sales_channel_l1_name NOT LIKE 'Online' THEN fbi.price_total_excl_vat
          ELSE 0
        END
      ) AS retail_sales_53w_104w
    FROM
      odl.fact_basket_items fbi
      LEFT JOIN odl.dim_customers dc USING (customer_sk)
      LEFT JOIN odl.dim_products USING (product_sk)
      LEFT JOIN odl.dim_categories_fpna ON odl.dim_products.category_fpna_sk = odl.dim_categories_fpna.category_fpna_sk
      LEFT JOIN odl.dim_sales_channels USING (sales_channel_sk)
    WHERE
	  date_trading_nk < '2026-02-12'
      AND customer_nk IN (
        SELECT DISTINCT
          customer_nk
        FROM
          customers_campaign
      )
    GROUP BY
      1,
      2,
      3,
      4
  ),
  churned AS (
    SELECT
      dc.customer_nk,
      1 AS did_txn_last_12m
    FROM
      odl.fact_basket_items fbi
      JOIN odl.dim_customers dc USING (customer_sk)
    WHERE
      fbi.country_sk IN ('hbi|eu|nl', 'hbi|eu|be')
      AND fbi.date_trading_nk >= '2026-02-12'
	  AND fbi.date_trading_nk <= '2026-03-12'
    GROUP BY
      dc.customer_nk
  )
SELECT
  c.customer_nk,
  c.has_rfl,
  c.gender,
  c.country_sk,
  churn_group AS treatment_indicator,
  ('2026-02-12' - c.last_order_date) AS recency,
  c.total_transactions AS frequency,
  c.sales_ttl AS monetary_value,
  c.volume_ttl AS total_volume,
  ('2026-02-12' - c.first_transaction_date) AS length_of_relationship,
  c.online_sales,
  c.retail_sales,
  c.food_total,
  c.vhms_total,
  c.sports_total,
  c.beauty_total,
  c.num_orders_52wk AS frequency_52wk,
  c.sales_52wk AS monetary_value_52wk,
  c.volume_52wk AS volume_52wk,
  c.online_sales_52w,
  c.retail_sales_52w,
  c.num_orders_53w_104w AS frequency_53w_104w,
  c.sales_53w_104w AS monetary_value_53w_104w,
  c.volume_53w_104w,
  c.online_sales_53w_104w,
  c.retail_sales_53w_104w,
  COALESCE(ch.did_txn_last_12m, 0) AS reactivated
FROM
  covariates_churn c
  LEFT JOIN churned ch ON ch.customer_nk = c.customer_nk
  LEFT JOIN customers_campaign ec ON c.customer_nk = ec.customer_nk