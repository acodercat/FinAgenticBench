# Adjudication items

For each item, decide which candidate value the task as worded requires for the named output. You may use the data sources. Mark A, B, or 'both' if the wording supports both readings, and say in one line why.

## Q01

**Data sources:** sec_nmfp_series_filings, sec_nmfp_liquidity, nyfed_reference_rates

**Turn 1.** Build a hypothetical USD10 million treasury liquidity vehicle with the balance-sheet mix of the three largest Government-category series in original Form N-MFP3 reports dated 2025-10-31. Require positive net assets, positive weighted-average maturity, and same-report-date daily and weekly liquid-asset amounts satisfying positive daily assets, weekly assets at least daily assets, and weekly assets no greater than net assets. Rank by net assets descending, breaking ties by SEC series ID ascending; weight the selected series by net assets. For this vehicle, the reported daily-liquid fraction becomes available at day 1, the additional weekly-liquid fraction at day 5, and the remaining assets at day 30, each at par. Treat these as a stipulated asset-availability ladder with no investment income. Report the largest selected series ID, blended weighted-average maturity and day-1 available fraction. Carry the vehicle’s complete composition forward and assess the scope of this construction. Use unrounded inputs and state throughout.

**Output to adjudicate (turn 1):** `government_daily_liquidity_pct`: Store Government day1 available asset share in percent, rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.

**Candidate A:** `62.9738`

**Candidate B:** `61.2814`

## Q02

**Data sources:** sec_financial_statements, sec_submissions

**Turn 1.** From the financial statements rather than the filing index, take full-year Form 10-K reports with a period end in calendar 2024, read from the report date rather than the fiscal-year label, excluding amended filings. Where a filer closes two fiscal years in that year, keep the later period end, then the later filing date, then the higher accession. Report how many filers this leaves, how many close their year on a date other than 31 December, how many report a year-to-date net income, and how many report a net income for the quarter alone. Explain what the last two counts say about where a fourth-quarter net income can and cannot be read from.

**Turn 2.** Pair each retained filer with its own third-quarter report: the Form 10-Q whose fiscal period is the third quarter, again excluding amendments, whose period end falls before the annual period end and no more than 120 days before it, taking the nearest such filing where more than one qualifies. Report how many filers pair, how many of those carry a year-to-date net income on both filings, the sum of the annual year-to-date net income across that group, and the sum across the same group of the annual figure less the nine-month figure. Explain what the second sum measures, and which filers the 120-day limit is there to keep out.

**Turn 3.** Stay with the filers whose fourth quarter you derived. Report how many report a positive year, how many of those have a negative derived fourth quarter, how many filers carry a quarterly net income on the annual filing alongside the derived one, and how many of that last group have a derived and a reported figure differing by a dollar or more. Explain how a profitable year ending in a loss-making quarter would show up in the year-to-date series itself, and what the size of the disagreeing group says about treating the subtraction as exact.

**Turn 4.** Look at the filers whose derived and reported quarters disagree. Count those whose reported quarterly net income equals their own annual year-to-date figure. Then identify the filer whose two figures sit furthest apart in absolute terms, breaking a tie by the lower CIK, and report that filer's CIK, its derived fourth quarter and its reported one. Explain what a filer reporting its annual amount in the quarterly field would indicate, and what could move the two figures apart where that is not what happened.

**Turn 5.** Check what the derived figure rests on. Join the annual filings to the submission index on filer and accession together; a jointly filed accession appears there once per filer. Report how many annual filings carry an accession that more than one filer files under, how many are later superseded by an amendment with the same period end and a later filing date, the median days from period end to public acceptance, and how many the two files label with different forms. Explain what the supersession count and the publication lag each do to a fourth-quarter figure derived as you derived it.

**Output to adjudicate (turn 5):** `superseded_by_amendment_count`: Store the count later superseded by an amendment for the same period, as an integer, rounded to 0 decimals from unrounded source rows.

**Candidate A:** `574`

**Candidate B:** `575`

## Q03

**Data sources:** fema_nfip, fema_disasters

**Turn 1.** Build a retrospective Louisiana calendar-2020 claim cohort from the frozen NFIP snapshot. A claim qualifies if its five-digit county FIPS and loss date fall within at least one FEMA declaration area's inclusive incident interval, among Louisiana declarations with incident beginning in 2020 and incident type Hurricane or Flood. Keep claims with all three gross building, contents and ICC payment fields present and a positive sum, counting each claim once however many declaration areas cover it. Report distinct qualifying claims and their source gross payment sum.

**Turn 2.** Build a hypothetical insured daily loss ledger at 40% of those gross payments, each loss at midnight on its loss date. An occurrence is a 72-hour half-open window starting at midnight on any date from August 20 through October 27, 2020. The treaty has a USD 5m attachment per window, a USD 10m layer, USD 10m of paid reinstatement capacity and a USD 18m annual aggregate. A window recovers its loss above attachment up to the current layer and remaining aggregate. After every settlement, exhausted layer is restored from unused reinstatement capacity, up to USD 10m, for cash premium of USD 3m times restored capacity over USD 10m times days from window end to 2021-01-01 over 366; premium leaves the aggregate untouched. For fixed starts August 25 and October 8, report total gross recovery and paid reinstatement premium. Explain their effect on later capacity.

**Output to adjudicate (turn 2):** `fixed_windows_gross_recovery_mn`: Store gross treaty recovery for the two fixed windows in USD millions, rounded to 4 decimals, calculated from unrounded inputs and not from other reported rounded values.

**Candidate A:** `10.0`

**Candidate B:** `18.0`

## Q04

**Data sources:** sec_midas_security_exchange, finra_short_volume

**Turn 1.** Using SEC MIDAS individual security-by-exchange observations and FINRA consolidated daily short-volume files for April 2025, determine their common trading dates. For cross-source ticker identity, match a MIDAS dot-form share-class ticker to the FINRA symbol obtained by replacing its period separator with a slash; otherwise require exact ticker-string equality. A ticker enters the comparison only if, on every common date, it has at least one MIDAS observation with both trade and odd-lot volume reported and one FINRA row. Report the common-date count and the eligible-ticker count. Carry that population forward and explain what the two sources each cover.

**Output to adjudicate (turn 1):** `full_coverage_common_ticker_count`: Store the full-common-date-coverage ticker count as an integer.

**Candidate A:** `6516`

**Candidate B:** `6180`

## Q05

**Data sources:** ken_french_10_industry_portfolios, ken_french_daily_factors, eia_bulk_wti_daily

**Turn 1.** Align the ten value-weighted industry portfolios with the daily research factors and the daily Cushing WTI spot price by date, computing a daily oil return as the percentage change in the spot price and keeping only dates where all three report. Report how many such dates fall in 2015-01-01 through 2019-12-31 and how many fall in 2020-02-19 through 2020-03-23, and report the summed market excess return over the second window in percent. Then, on the first window, regress each industry's daily return minus the risk-free rate on a constant and the market excess return. Report the label of the industry with the highest slope, that slope, and the lowest slope in the cross-section, slopes. Explain what the fitted coefficient captures about day-to-day co-movement over the sample used, and how far it carries to a later stress window.

**Turn 2.** Use each fitted line to predict the crisis window. Because the model is linear in daily returns, form the prediction as the intercept times the number of crisis days plus the slope times the summed market excess return, and compare it against the arithmetic sum of the industry's daily returns minus the risk-free rate over the same days rather than against a compounded return. Report the predicted and realized figures for utilities and for technology, each in percent. Then report the mean absolute difference between predicted and realized across all ten industries in percentage points, and how many industries realized a lower figure than predicted. Explain what the utilities comparison says about using a low estimated beta as crisis protection.

**Turn 3.** Scale the misses relative to the fitted model's estimation-period residual volatility. For each industry take the standard deviation of its estimation-period residuals and scale it to the crisis window by multiplying by the square root of the number of crisis days, then divide each industry's prediction error by that figure. Report the resulting number of scaled residual standard deviations for utilities, for technology and for telecommunications, and how many of the ten industries lie more than three such units from their prediction. Then report the label of the industry with the largest absolute error and that industry's signed prediction error, actual minus predicted, in percentage points. Explain why the raw-error and standardized-error rankings can differ and state the statistical limitations of this scaling.

**Output to adjudicate (turn 3):** `utilities_standard_errors`: Store the utilities error divided by its window standard error, rounded to 4 decimals.

**Candidate A:** `-7.754`

**Candidate B:** `-7.7509`

## Q06

**Data sources:** eia_bulk_wti_daily, ken_french_10_industry_portfolios, ken_french_daily_factors, ofr_market_stress

**Turn 1.** On the full governed WTI price history, first form log returns as 100 times the adjacent-date log price ratio, retaining a return only when both prices are positive; then apply the 2018-2025 return-date window. Align those returns with Kenneth French Energy-industry and risk-free returns and the OFR Financial Stress Index, with Energy excess returns in percentage points. Split 2018-2022 for estimation and 2023-2025 for conditional evaluation. Report all three counts; across the aligned panel select the maximum OFR-stress date, breaking a tie by earlier date, and report its OFR index, Energy excess return and WTI return plus the minimum aligned WTI price. Carry the aligned panel and split forward and interpret the selected cross-market episode and its limits.

**Turn 2.** On the carried 2018-2022 sample, estimate a deterministic two-start local Gaussian maximum-likelihood oil-conditioned GARCH(1,1): an intercept-bearing Energy-excess conditional mean with contemporaneous WTI log return, and a variance recursion with positive omega, nonnegative shock and lagged-variance coefficients, and their sum strictly below 0.999. Initialize the first variance at the preliminary OLS residual variance using n-1. Optimize an exponential-omega/stationary-softmax parameterization by L-BFGS-B to a 0.0000000000001 relative objective tolerance and 3000-iteration cap. Both starts use the preliminary OLS mean coefficients; their omega values are 0.03 and 0.08 times the initial variance and their shock/lagged-variance pairs are respectively (0.0158603637364903, 0.8659465188651376) and (0.0899405425972101, 0.6645757148190471). Choose the converged finite fit with greater likelihood, breaking an exact tie lexicographically; fail if neither converges. Report the mean coefficients, variance parameters, persistence, implied unconditional volatility, maximized log likelihood, and the maximum absolute unscaled squared-percentage recursion residual over observations after the initializer. Also report the minimum fitted-minus-perturbed log-likelihood loss when, holding the fitted mean and initializer fixed, omega is multiplied by 1.05 or 0.95, alpha by 1.01, or beta by 0.999 one at a time. Carry the fitted state forward and interpret persistence, conditioning and model-distribution limits.

**Output to adjudicate (turn 2):** `garch_mean_intercept_pct`: Store the conditional-mean intercept in percent, rounded to 5 decimals.

**Candidate A:** `0.04277`

**Candidate B:** `0.02363`

## Q07

**Data sources:** treasury_auctions, treasury_yield_curve

**Turn 1.** From Treasury bill auctions issued 2023-01-01 through 2025-12-31, excluding cash management bills, report count. For bills with issue-to-maturity terms of at most 182 days, infer the calendar-year day count that reproduces each published high investment rate to three decimals from a simple return on its published high price per USD 100; report cohort count. Apply the inferred convention to all selected bills and report how many use 366 days. Explain the day-count rule and the distinction between published high discount and investment rates.

**Output to adjudicate (turn 1):** `leap_basis_count`: Store how many auctions take a 366-day year under the reproducing rule, as an integer.

**Candidate A:** `273`

**Candidate B:** `258`

## Q08

**Data sources:** fdic_bankfind, nyfed_reference_rates

**Turn 1.** Use FDIC BankFind quarterly reports from 2021 Q1 through 2024 Q4. Keep the balanced bank panel with a nonmissing strictly positive reported net interest margin in every quarter, identifying banks by FDIC certificate. Use the reported annualized margin in percent. The rate regressor is the equal-fixing arithmetic mean of published New York Fed SOFR effective-date observations within each quarter, in percent. Fit a first-order dynamic individual-effects margin model with its own one-quarter lag and contemporaneous SOFR, constant slopes and no time effects. Use one-step Arellano–Bond difference GMM with homoskedastic differenced-error weighting. For the lagged dependent variable, use available level lags two through four only, collapsed by lag; there are no pre-2021 observations, and each differenced equation from 2021 Q3 onward retains its available instruments. Treat SOFR as strictly exogenous with one ordinary-IV instrument column containing its own differenced value. Include no level-equation moments or differenced intercept, and leave coefficients unrestricted. Report bank count and both coefficients. Carry the complete panel, instrument specification and fitted residuals forward, and interpret the estimated margin dynamics. Use unrounded inputs and state throughout.

**Output to adjudicate (turn 1):** `one_step_persistence`: Store one-step lagged-margin coefficient, dimensionless rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.

**Candidate A:** `0.632`

**Candidate B:** `0.6281`

## Q09

**Data sources:** sec_nport, ken_french_daily_factors

**Turn 1.** Form N-PORT reports three months of share sales, share redemptions and investment gains for each filing, together with net assets at the period end. Work only with filings that report a fund series identifier. Keep, for each series and period end, only the filing carrying the latest filing date, taking that filing whole rather than filling its gaps from the one it supersedes, and breaking a tie by the later accession number. Then keep a filing only where all three months of sales, all three of redemptions, both gain components in all three months and net assets are reported, and where net assets are above zero. Report how many filings remain, how many distinct series they cover, the total amount redeemed across every month they report, and the total amount sold across the same months, the last two in billions of dollars.

**Output to adjudicate (turn 1):** `total_gross_redeemed_usd_billions`: Store the total amount redeemed across every month they report in billions of dollars, rounded to 4 decimals.

**Candidate A:** `53220.9308`

**Candidate B:** `35417.3573`

## Q10

**Data sources:** sba_504, sba_7a

**Turn 1.** Construct two retrospective hypothetical bank books from California project rows in the frozen June 30, 2026 SBA snapshot. Keep fiscal-year 2024 approvals whose recorded first disbursement is on or before December 31, 2024. For 504 require positive SBA gross approval and third-party dollars; for 7(a) require positive gross approval and SBA-guaranteed approval. Keep the reported-row grain. In 504 the bank owns only the senior third-party claim, and the SBA gross approval is a junior claim. Scale each program's dollar amounts uniformly so that its bank principal, the 504 senior claim or the 7(a) gross loan, totals USD 100 million. Report the two retained row counts and normalized 504 junior principal.

**Turn 2.** Give each reported row collateral: for 504, total senior and junior principal at a 90% combined loan-to-value; for 7(a), 110% of gross principal. Every row defaults immediately, and recovery is 60% of initial collateral less collection costs of 5% of the gross recovered cash. The 504 senior claim has first priority, the junior claim takes only the residual, and neither recovers more than its principal. The 7(a) bank has sole priority over its collateral; ignore guarantee payments in this turn, accrued interest and all other recoveries. Report 504 senior and junior principal losses and the 7(a) bank collateral shortfall. Explain how claim priority affects the losses.

**Turn 3.** Apply the following hypothetical 7(a) guarantee to that bank shortfall: each row's coverage fraction is its reported guaranteed approval divided by gross approval, and this fraction covers the net principal shortfall after collection costs. The guarantee pays exactly 24 months after the immediate collateral recovery, with no fees, disputes or further collateral recoveries. Discount the guarantee at a 5% effective annual rate. Report bank 7(a) present-value principal loss in USD millions and select the lower-loss bank book. Explain the effect of guarantee coverage and payment timing.

**Output to adjudicate (turn 3):** `deterministic_lower_loss_book`: Store the lower-loss bank book as the exact string SBA504 or SBA7A, with SBA504 breaking an exact tie.

**Candidate A:** `"SBA504"`

**Candidate B:** `"SBA7A"`

## Q11

**Data sources:** nyfed_reference_rates, treasury_yield_curve

**Turn 1.** Treat New York Fed SOFR observations from 2021-01-01 through 2024-12-31 as a short-rate proxy and estimate the constant-parameter Vasicek model under the physical measure by exact Gaussian transition maximum likelihood, conditional on the first observation. Each consecutive published observation is one interval of 1/252 model-year. Require positive mean-reversion speed and diffusion volatility, with an unrestricted long-run level. Report transition count, mean-reversion speed, long-run level and diffusion volatility. Carry the fitted dynamics and final observed short rate forward and discuss their financial scope. Use unrounded state throughout.

**Turn 2.** Retain the carried mean-reversion speed, diffusion volatility and current short rate for risk-neutral Vasicek pricing. Calibrate only the risk-neutral long-run level so that a hypothetical five-year regular semiannual bond prices at par on 2024-12-31; its annual coupon is that date's Treasury five-year constant-maturity par quote. Use 100 principal, ten half-year payments and zero accrued interest. Report the fitted risk-neutral long-run level and five-year discount factor. Carry the pricing model forward and interpret the calibration relative to the physical estimate.

**Output to adjudicate (turn 2):** `risk_neutral_theta_pct`: Store the fitted risk-neutral long-run short rate in percent, rounded to 4 decimals. Compute from the unrounded carried model or source inputs.

**Candidate A:** `4.1262`

**Candidate B:** `-2.2737`

## Q12

**Data sources:** sba_504, sba_7a

**Turn 1.** Construct two retrospective hypothetical bank books from California project rows in the frozen June 30, 2026 SBA snapshot. Keep fiscal-year 2024 approvals whose recorded first disbursement is on or before December 31, 2024. For 504 require positive SBA gross approval and third-party dollars; for 7(a) require positive gross approval and SBA-guaranteed approval. Keep the reported-row grain. In 504 the bank owns only the senior third-party claim, and the SBA gross approval is a junior claim. Scale each program's dollar amounts uniformly so that its bank principal, the 504 senior claim or the 7(a) gross loan, totals USD 100 million. Report the two retained row counts and normalized 504 junior principal.

**Turn 2.** Give each reported row collateral: for 504, total senior and junior principal at a 90% combined loan-to-value; for 7(a), 110% of gross principal. Every row defaults immediately, and recovery is 60% of initial collateral less collection costs of 5% of the gross recovered cash. The 504 senior claim has first priority, the junior claim takes only the residual, and neither recovers more than its principal. The 7(a) bank has sole priority over its collateral; ignore guarantee payments in this turn, accrued interest and all other recoveries. Report 504 senior and junior principal losses and the 7(a) bank collateral shortfall. Explain how claim priority affects the losses.

**Turn 3.** Apply the following hypothetical 7(a) guarantee to that bank shortfall: each row's coverage fraction is its reported guaranteed approval divided by gross approval, and this fraction covers the net principal shortfall after collection costs. The guarantee pays exactly 24 months after the immediate collateral recovery, with no fees, disputes or further collateral recoveries. Discount the guarantee at a 5% effective annual rate. Report bank 7(a) present-value principal loss in USD millions and select the lower-loss bank book. Explain the effect of guarantee coverage and payment timing.

**Turn 4.** Replace deterministic recovery by two equally likely common gross-recovery states, 35% and 85%, applied to every row. Retain all collateral, priority, costs and guarantee terms. Report expected bank 504 principal loss, expected bank 7(a) present-value principal loss, the lower expected-loss bank book and worst-state bank 504 loss in USD millions. Explain the relationship between this result and the earlier deterministic comparison.

**Output to adjudicate (turn 4):** `stochastic_504_worst_loss_mn`: Store worst-state bank 504 principal loss in USD millions, rounded to 4 decimals.

**Candidate A:** `47.3979`

**Candidate B:** `35.0591`

## Q13

**Data sources:** sec_form_d, treasury_yield_curve

**Turn 1.** Use Seven Hills Capital Fund I LP's original Form D accession 0000895527-24-000002 as a numeric template for a new hypothetical fund. LPs contribute its reported total sold amount on January 1, 2025; reported sales commissions plus finder fees define the expense paid then. Allocate contribution, expense and remaining investment capital across three projects at 40%/35%/25%. Each project exits completely on January 1 of 2027/2028/2030, returning respectively 1.8/1.4/0.4 times its invested capital. There are no other flows, taxes or GP contributions. Report total LP contribution and total investable capital. Carry unrounded state between turns.

**Turn 2.** Apply a whole-of-fund waterfall to those exits: first return contributed capital, then pay accumulated preferred return of 8% simple annually, ACT/365, on unreturned contributed capital; unpaid preferred return earns no interest. Next give incremental cash entirely to the GP until its cumulative distributions equal 20% of cumulative distributed profit; preferred return counts as profit and returned capital does not. Thereafter split residual cash 80% LP/20% GP. Fees are included in contributed capital. Report LP and GP cash at the second exit and cumulative terminal GP entitlement. Explain the effect of distribution priority at the second exit.

**Turn 3.** Now apply the same waterfall separately to each project, using its allocated contribution and the time since its call, with carry paid at its exit. At final settlement the GP's cumulative entitlement is the whole-of-fund waterfall on the same gross project cash history: any prior overpayment returns to LPs at once and any remaining entitlement goes to the GP, who can always repay. Discount on the 2024-12-31 Treasury yields for 1, 2, 3, 6 months and 1, 2, 3, 5, 7, 10, 20, 30 years, read as zero rates compounded semiannually, interpolated linearly in years with flat ends, on ACT/365 with no credit spread. Report carry paid before the final exit, the present value at the initial call of all LP cash including true-up, and the contract with the higher LP present value. Explain how carry timing bears on the comparison.

**Output to adjudicate (turn 3):** `early_deal_carry_mn`: Store carry actually paid at the first two deal-by-deal exits in USD millions, rounded to 4 decimals.

**Candidate A:** `10.9671`

**Candidate B:** `11.2974`

## Q14

**Data sources:** bis_global_liquidity, ofr_market_stress

**Turn 1.** Build a quarterly panel with BIS total credit (loans and debt securities, all lenders) to non-bank borrowers in emerging-market and developing economies for USD, EUR, and JPY, plus the calendar-quarter arithmetic mean of daily OFR Financial Stress Index values. Compute each currency's quarter-over-quarter log growth times 100 using the immediately preceding calendar-quarter level from the full BIS series, then retain transformed observations from 2001-Q1 through 2024-Q4. Order the four variables USD growth, EUR growth, JPY growth, OFR FSI. Estimate an intercept-bearing unrestricted VAR(2) by equation-by-equation OLS. Estimate innovation covariance from the residual cross-product divided by the number of VAR regression observations, not residual degrees of freedom. Report panel and regression counts, companion spectral radius, 2-norm condition numbers of the intercept-and-lag design and innovation covariance, and the maximum absolute sample normal-equation moment. Explain the statistical status of the stationarity and conditioning diagnostics, discuss the material scope limitations of combining the four series in one fitted system, and carry the VAR and innovation covariance forward.

**Turn 2.** Using the carried VAR, compute the Pesaran-Shin generalized forecast-error variance decomposition at an eight-quarter forecast horizon, using moving-average horizons zero through seven and the full innovation covariance. Row-normalize the resulting decomposition. Report the complete 4-by-4 matrix in percent, with each requested name read as source-to-target, plus the maximum absolute normalized row-sum residual and the Diebold-Yilmaz total connectedness index. Explain the distinction between generalized shares and orthogonalized Cholesky shares and the interpretive scope of the row normalization.

**Turn 3.** From the carried row-normalized decomposition, for each node report directional connectedness received from other nodes and transmitted to other nodes, and compute transmitted minus received net connectedness. Select the greatest net transmitter, ties in the stated node order. Also identify the largest off-diagonal directed edge, ties by target order then source order, and report source, target, and share. Identify every reported node using exactly one lower-case token from usd, eur, jpy, or ofr. Interpret the network direction and discuss what these inferred spillovers establish about transmission and causality.

**Turn 4.** Shock the carried net-transmitter node by one positive innovation standard deviation and compute its Pesaran-Shin generalized impulse response over horizons zero through eight using the full carried innovation covariance. Report the shock node using the same exact lower-case node token, its size, OFR-FSI impact response, the earliest horizon with greatest absolute OFR response and its signed value, cumulative responses over all nine horizons for each ordered node, and the horizon-eight OFR response. Explain how correlated contemporaneous innovations affect a generalized impulse response and the evidentiary limits of interpreting this estimated shock response as a forecast or policy experiment.

**Output to adjudicate (turn 4):** `girf_ofr_peak_absolute_horizon`: Store the horizon from zero through eight with greatest absolute OFR-FSI response as an integer.

**Candidate A:** `0`

**Candidate B:** `4`

## Q15

**Data sources:** ffiec_call_reports, treasury_yield_curve

**Turn 1.** Build a hypothetical IFRS acquisition from the 2025-09-30 consolidated Call Reports of buyer RSSD 197478 and target RSSD 141556, treated as independent banks. Book equity is equity attributable to a bank's own shareholders, excluding inherited subsidiary NCI. The buyer pays 1.4 times the book equity of an 80% stake. Appraise the other 20%: 5% of its book equity at years 1 to 5 plus 1.4 times its book equity at year 5, discounted at the 2025-09-30 Treasury yields for 1, 2, 3, 6 months and 1, 2, 3, 5, 7, 10, 20, 30 years read as zero rates, semiannually compounded, linearly interpolated with flat ends. Report consideration and appraised NCI.

**Output to adjudicate (turn 1):** `nci_fair_value_mn`: Store independently appraised noncontrolling interest in USD millions, rounded to 4 decimals, calculated from unrounded inputs and not from other reported rounded values.

**Candidate A:** `744.3721`

**Candidate B:** `148.8744`

## Q16

**Data sources:** sec_financial_statements, sec_submissions

**Turn 1.** From the financial statements rather than the filing index, take full-year Form 10-K reports with a period end in calendar 2024, read from the report date rather than the fiscal-year label, excluding amended filings. Where a filer closes two fiscal years in that year, keep the later period end, then the later filing date, then the higher accession. Report how many filers this leaves, how many close their year on a date other than 31 December, how many report a year-to-date net income, and how many report a net income for the quarter alone. Explain what the last two counts say about where a fourth-quarter net income can and cannot be read from.

**Turn 2.** Pair each retained filer with its own third-quarter report: the Form 10-Q whose fiscal period is the third quarter, again excluding amendments, whose period end falls before the annual period end and no more than 120 days before it, taking the nearest such filing where more than one qualifies. Report how many filers pair, how many of those carry a year-to-date net income on both filings, the sum of the annual year-to-date net income across that group, and the sum across the same group of the annual figure less the nine-month figure. Explain what the second sum measures, and which filers the 120-day limit is there to keep out.

**Turn 3.** Stay with the filers whose fourth quarter you derived. Report how many report a positive year, how many of those have a negative derived fourth quarter, how many filers carry a quarterly net income on the annual filing alongside the derived one, and how many of that last group have a derived and a reported figure differing by a dollar or more. Explain how a profitable year ending in a loss-making quarter would show up in the year-to-date series itself, and what the size of the disagreeing group says about treating the subtraction as exact.

**Turn 4.** Look at the filers whose derived and reported quarters disagree. Count those whose reported quarterly net income equals their own annual year-to-date figure. Then identify the filer whose two figures sit furthest apart in absolute terms, breaking a tie by the lower CIK, and report that filer's CIK, its derived fourth quarter and its reported one. Explain what a filer reporting its annual amount in the quarterly field would indicate, and what could move the two figures apart where that is not what happened.

**Turn 5.** Check what the derived figure rests on. Join the annual filings to the submission index on filer and accession together; a jointly filed accession appears there once per filer. Report how many annual filings carry an accession that more than one filer files under, how many are later superseded by an amendment with the same period end and a later filing date, the median days from period end to public acceptance, and how many the two files label with different forms. Explain what the supersession count and the publication lag each do to a fourth-quarter figure derived as you derived it.

**Output to adjudicate (turn 5):** `median_days_to_public`: Store the median days from period end to public acceptance, rounded to 1 decimals from unrounded source rows.

**Candidate A:** `65.9`

**Candidate B:** `65.0`

## Q17

**Data sources:** fed_z1_vintages, ken_french_daily_factors

**Turn 1.** Work in the newest release of the Financial Accounts only. Corporate equities on the nonfinancial corporate business balance sheet appear under quarterly series codes sharing the digits 103064103, one beginning LM, one FA, one FU and one FR. Report how many quarter-over-quarter changes the instrument supports, its market value at the final quarter end, the net issuance the sector transacted summed across those quarters, and how many of those quarters carry a negative net issuance. Explain what a negative net issuance says about what the sector was doing.

**Output to adjudicate (turn 1):** `net_retirement_quarter_count`: Store how many of those quarters carry a negative net issuance as an integer.

**Candidate A:** `33`

**Candidate B:** `34`

## Q18

**Data sources:** hmda, treasury_yield_curve

**Turn 1.** Use the frozen 2024 HMDA register's California conventional first-lien originations for owner-occupied, one-unit properties, retaining numeric interest rates from 0.1 through 20 percent inclusive. Report the retained count and unweighted median interest rate. Estimate a normal short-rate diffusion volatility from consecutive one-year Treasury CMT yield changes wholly within calendar 2024, using sample standard deviation, one trading day per consecutive observation and 252 trading days per year. Report the increment count and estimated volatility. Carry the median rate and unrounded volatility forward and discuss their financial scope.

**Turn 2.** Build a four-year semiannual analytical discount curve from Treasury's 2024-12-31 par yields at six months and one, two, three and five years, with linear maturity interpolation of par yields and semiannual par-instrument bootstrapping. Calibrate a recombining additive Ho–Lee short-rate tree to that complete discount curve using the carried volatility, half-year steps, equal risk-neutral branch probabilities and continuously compounded node rates without a rate floor. Report the four-year discount factor and the lowest short rate at year two. Carry the tree forward and interpret the resulting discounting and short-rate profile.

**Turn 3.** Value a hypothetical loan of 100 initial principal with the carried median as its nominal annual coupon, eight equal semiannual total payments and full amortization at year four. The borrower can repay the remaining scheduled principal immediately after each of the first seven payments, with no fees or default; there is no inception prepayment. Assume optimal borrower exercise, with prepayment at exact indifference. Using the carried Ho–Lee tree, report its nonprepayable and prepayable values at zero spread. Then fit a constant continuously compounded option-adjusted spread to a price of 95, including that spread in both discounting and the borrower's continuation valuation, and report the spread. Carry the fitted spread and optimal policy forward and interpret the option cost and spread.

**Turn 4.** Under the carried fitted-spread policy, report the risk-neutral probability of prepayment strictly before maturity. Holding the coupon, payment grid, volatility and fitted spread fixed, measure standard symmetric effective duration and convexity from parallel one-basis-point upward and downward shifts to every continuously compounded zero rate on the base analytical discount curve, with the Ho–Lee tree recalibrated and exercise reoptimized for each shift. Report both sensitivities and interpret the resulting risk profile.

**Output to adjudicate (turn 4):** `prepayment_probability`: Store the pre-maturity prepayment probability as a decimal probability, rounded to 4 decimals. Compute from its own unrounded source inputs or carried model state, not from other rounded outputs.

**Candidate A:** `0.0156`

**Candidate B:** `0.0`

## Q19

**Data sources:** treasury_auctions, nyfed_reference_rates

**Turn 1.** Identify the U.S. Treasury floating-rate note auctioned in March 2025. The auction record does not carry a floating-rate security type, so use the field that flags a floating rate rather than the security-type label. If more than one qualifies, take the latest auction date and break a remaining tie by CUSIP ascending. Buy USD 1,000,000 of face value at that auction's published price and published accrued interest, with no fees or haircut. Report the CUSIP, the settlement date, the first interest payment date after settlement, and the cash required at settlement in USD thousands to five decimals, the precision a cent-exact amount needs in that unit. Settle every cash amount to the nearest cent with half a cent rounded up, and carry the note, its dates and the settlement cash forward. Explain what the accrued interest represents in the buyer's settlement cash.

**Turn 2.** Finance the whole settlement cash at realized SOFR with no additional spread. A date carrying a published SOFR observation is a business day; one simple ACT/360 factor covers each interval between consecutive observations, including any non-business days inside it, and the factors compound across intervals. The note's coupon accrues daily from its dated date on an index equal to the high discount rate of the most recent thirteen-week bill auction, converted to an ACT/360 money-market yield over that bill's own term from issue to maturity, plus the note's spread. An index takes effect only on days after the auction that set it, and the index in force is held fixed for the final two business days before each payment date. Round the converted index and each day's interest per USD 100 of face to nine decimals. At the first payment, accrue financing through but excluding the payment date and apply the entire coupon to the loan. Report the first coupon received and the loan balance immediately after that payment, both in USD thousands to five decimals. Explain how the period the coupon covers differs from the period the buyer has funded.

**Turn 3.** Hold the same face amount through the following quarterly payment date, which follows the note's month-end dated date rather than falling three calendar months after the first payment. Continue financing the outstanding loan under the same conventions and again apply the whole coupon on its payment date. Report that payment date, its coupon in USD thousands, and the loan balance immediately after repayment in USD thousands, both to five decimals. Discuss what this balance establishes about the funded position, and what further observation would be needed before it could be read as a return.

**Output to adjudicate (turn 3):** `second_coupon_usd_thousands`: Store the second coupon received in USD thousands as a float, rounded to 5 decimals.

**Candidate A:** `11.21522`

**Candidate B:** `11.21493`

## Q20

**Data sources:** sec_notes_dimensional, treasury_yield_curve

**Turn 1.** Use 10-K single-axis notes facts on the retirement plan type axis's defined benefit pension member, primary presentation (index 0), each from its earliest filing (tie: lower accession); an item's year end is its latest fact date in the calendar year. Take filers with a benefit-obligation discount rate at both 2021 and 2022 year ends: report their count and how many have a rate outside (0, 1) at either date, then drop those. For the rest report the median rate change. Medians use linear interpolation throughout. Explain how sponsors set this rate.

**Turn 2.** Compare the rest with the ten-year Treasury par yield on each year-end date, or on the latest earlier date with one. Report the median change in that yield between the two year ends, how many filers' rate changes lie within 50 basis points, inclusive, of that same filer's own yield change, and, at the 2022 year end, the median rate, how many rates stand strictly below the yield, and the median of rate less yield. Explain how the rate moves with the Treasury yield and where it stands against it.

**Turn 3.** Of these filers, take those carrying the actuarial gain or loss for the year ending at their 2022 year end and the benefit obligation at both year ends. Report how many there are, how many tag the gain below zero, and how many of those show a lower obligation in 2022 than in 2021. Then divide each filer's gain size, whatever its sign, by its 2021 obligation and by the rate change as a decimal, and report the median in years. Explain what the rate change did to the obligation, what this ratio measures, and what to make of a gain tagged below zero.

**Output to adjudicate (turn 3):** `median_implied_duration_years`: Store the median implied duration in years, rounded to 4 decimals from unrounded source rows.

**Candidate A:** `9.0457`

**Candidate B:** `9.1609`

## Q21

**Data sources:** ken_french_10_industry_portfolios, ken_french_daily_factors

**Turn 1.** Use Kenneth French's value-weighted HiTec, Utils, Enrgy and NoDur daily portfolios and the daily risk-free return on common dates in 2018-2022. Estimate the covariance of daily decimal excess returns with the sample denominator and annualize it with 252 observations. For a stipulated equal-capital equilibrium benchmark and standard Markowitz risk aversion four on annual decimal returns, obtain the CAPM reverse-optimization equilibrium mean vector. Report the training count, HiTec's implied annual excess mean, the benchmark's annual volatility and the covariance spectral condition number. Carry the covariance and full equilibrium mean vector forward and discuss the prior's financial meaning.

**Output to adjudicate (turn 1):** `equilibrium_hitec_mean_pct`: Store the implied HiTec annual excess mean in percent, rounded to 4 decimals. Compute from its own unrounded source inputs or carried model state, not from other rounded outputs.

**Candidate A:** `19.4344`

**Candidate B:** `19.4189`

## Q22

**Data sources:** sec_financial_statements, sec_companyfacts

**Turn 1.** Select original full-year Form 10-Ks ending in calendar 2024, one per filer by latest report date, filing date, then accession. Keep filers reporting fiscal-year values for all four of operating cash flow, PP&E purchases, common-stock repurchases and share-based compensation. Exclude filers reporting a negative value for any of the last three concepts. Report retained filer count, excluded count, positive-repurchase count and total retained operating cash flow. Explain what an unexpected negative value can and cannot establish and whether reversing its sign is justified.

**Output to adjudicate (turn 1):** `negative_value_excluded_filer_count`: Store the count dropped for a negative outflow or expense, as an integer, rounded to 0 decimals from unrounded source rows.

**Candidate A:** `0`

**Candidate B:** `8`

## Q23

**Data sources:** bis_global_liquidity, ofr_market_stress

**Turn 1.** Build a quarterly panel with BIS total credit (loans and debt securities, all lenders) to non-bank borrowers in emerging-market and developing economies for USD, EUR, and JPY, plus the calendar-quarter arithmetic mean of daily OFR Financial Stress Index values. Compute each currency's quarter-over-quarter log growth times 100 using the immediately preceding calendar-quarter level from the full BIS series, then retain transformed observations from 2001-Q1 through 2024-Q4. Order the four variables USD growth, EUR growth, JPY growth, OFR FSI. Estimate an intercept-bearing unrestricted VAR(2) by equation-by-equation OLS. Estimate innovation covariance from the residual cross-product divided by the number of VAR regression observations, not residual degrees of freedom. Report panel and regression counts, companion spectral radius, 2-norm condition numbers of the intercept-and-lag design and innovation covariance, and the maximum absolute sample normal-equation moment. Explain the statistical status of the stationarity and conditioning diagnostics, discuss the material scope limitations of combining the four series in one fitted system, and carry the VAR and innovation covariance forward.

**Output to adjudicate (turn 1):** `maximum_ols_normal_moment`: Store the maximum absolute intercept/lag normal-equation moment, rounded to 10 decimals.

**Candidate A:** `0.0`

**Candidate B:** `3051.1177027997`

## Q24

**Data sources:** nyfed_reference_rates, treasury_yield_curve

**Turn 1.** The reference-rate table publishes 30-, 90- and 180-day SOFR averages. Take the publication dates from 2023-01-01 through 2025-12-31 carrying all three, and report how many there are. Each average covers the calendar days from that many days before the publication date to the day before it. Find how the daily SOFR rates are combined over a window so that your result, rounded half up to five decimals, reproduces every published average on those dates, and report its unrounded 30-day and 180-day averages for 2025-12-31. Take every median in this task with linear interpolation. Explain the calculation and why no simpler reading reproduces the published figures.

**Turn 2.** For every one of those publication dates and for the 30-day and 180-day windows, also compute the simple average in which every calendar day of the window carries the daily rate that applies to it, and take your own unrounded reproducing average, not the published five-decimal figure, less that simple average in basis points. Report the median of that difference for the 30-day window and for the 180-day window, and the largest difference for the 180-day window, taking medians with linear interpolation. Explain what the difference measures and what makes it larger.

**Output to adjudicate (turn 2):** `median_premium_30_day_bp`: Store the median 30-day compounded less simple average in basis points, rounded to 4 decimals from unrounded source rows.

**Candidate A:** `0.9102`

**Candidate B:** `0.9428`

## Q25

**Data sources:** sec_companyfacts, treasury_yield_curve

**Turn 1.** In the largest-filer company facts, use unamended annual 10-K reports and first-reported values (earliest filing, tie: lower accession). For each item, use its latest tagged balance date in 2024. Keep filers tagging total undiscounted operating lease payments, the interest portion and total lease liability on the same date. Report their count, how many reconcile exactly using tagged signs, how many tag negative interest, and the median interest as a percent of undiscounted payments across all selected filers. Use linear interpolation for every median in this task. Explain the liability and interest portion.

**Turn 2.** Keep the filers that reconcile exactly and tag the weighted-average operating lease discount rate, stored as a fraction, at the same 2024 balance date. Compare each with the five-year Treasury par yield on that date, or the latest earlier date with a yield. Report how many filers that is, their median rate, how many rates stand strictly below the yield, and the median of rate less yield. Explain what rate a lessee discounts its lease payments with, and what to make of a reported rate below the Treasury yield.

**Output to adjudicate (turn 2):** `rate_filer_count`: Store the count of reconciling filers tagging a discount rate as an integer.

**Candidate A:** `339`

**Candidate B:** `333`

## Q26

**Data sources:** fema_disasters, fema_nfip

**Turn 1.** Take the designated-area rows of federal disaster declarations whose incident began in calendar 2024, reading the incident begin date rather than the declaration date, with incident type Flood, Hurricane or Tropical Storm. Leave out rows whose county code ends in 000: they name tribal or statewide areas, several of which can share one code within a state. Report how many rows remain, how many distinct disasters and distinct counties they cover, and how many rows are major disaster declarations rather than emergencies. Explain what a single designated-area row records, and why the row count and the county count answer different questions.

**Turn 2.** Take the county as the unit and restrict county-level work to the major disaster declarations, a county carrying a programme when any of its designations in the window does. Report how many designated-area rows across the whole window carry the individuals and households programme, how many counties carry it, how many carry public assistance without it, and how many counties are designated under more than one of these major disaster declarations. Explain what it means for a county to sit inside a declaration and outside the household programme.

**Output to adjudicate (turn 2):** `ihp_county_count`: Store the count of counties carrying it on any designation as an integer, rounded to 0 decimals from unrounded source rows.

**Candidate A:** `3238`

**Candidate B:** `319`

## Q27

**Data sources:** eia_steo_vintages, eia_bulk_wti, nyfed_reference_rates

**Turn 1.** Establish a hypothetical WTI inventory-acquisition program from the EIA Short-Term Energy Outlook published on 2024-01-04. Use its forecast monthly WTI spot averages for May through December 2024. Allocate USD10 million of planned purchases to each month and fix that month’s barrel quantity using its forecast price; fractional barrels are allowed. Purchases occur at each calendar month-end and all inventory is retained through 2024-12-31. Report total planned inventory and the share scheduled for May and June together. Interpret the monthly volume distribution. Use unrounded inputs and state throughout.

**Turn 2.** Protect the original program’s monthly purchase cash requirements with hypothetical cash-settled swaps: receive that month’s PET.RWTC.M spot average and pay a common fixed price equal to the frozen December 2023 PET.RWTC.M observation plus USD2 per barrel. Each swap settles at its purchase month-end, covers a common fraction of that month’s fixed purchase quantity, and has no premium, fees or collateral cash flows. Choose the smallest common continuous hedge fraction between zero and 80 percent such that no month’s net purchase cash outlay exceeds USD11 million when every price is 25 percent above its original forecast. Report the fraction and the earliest target month whose cash limit binds. Commit the resulting monthly swap notionals for the original program and interpret the achieved protection.

**Turn 3.** Prepare an independent proposal using the EIA forecast published on 2024-04-04 for the same purchase months. Give it the same monthly planned budget, determine its own quantities, and choose its own minimum common hedge fraction under the same swap strike, participation bounds and monthly cash ceiling, with the 25-percent upside scenario relative to its own forecasts. Report proposal inventory and hedge fraction. Compare the proposal's barrels and hedge share with the committed plan's, and explain what in its own forecasts and the unchanged strike drives the difference.

**Turn 4.** For that proposal, settle its purchases and swaps using the frozen PET.RWTC.M observations for the purchase months. Report total purchase spending net of swap settlements, and gross swap settlement payments, counting receipts separately rather than offsetting them in the gross payment amount. Assess these execution costs in relation to the proposal’s delivery scale.

**Output to adjudicate (turn 4):** `proposal_gross_swap_payments_million`: Store proposal gross cash swap settlement payments in USD millions, rounded to 4 decimals. Compute from your own unrounded source inputs and carried state, never from other rounded outputs.

**Candidate A:** `27.468`

**Candidate B:** `0.6216`

## Q28

**Data sources:** sec_form_c, nyfed_reference_rates

**Turn 1.** For Aquipor Technologies, Inc. (CIK 0001765489), compare Form C/A accessions 0001665160-25-000665 and 0001665160-25-001224 for the same Regulation Crowdfunding offering. Report the earlier and revised offering deadlines, and from the June 18, 2025 amendment report the target and maximum offering amounts, most-recent-fiscal-year cash and equivalents, and short-term debt. Explain what the cash and short-term debt figures pulled from this amendment actually tell you about the company's position on the day it was filed.

**Turn 2.** Use the carried June 18 amendment's reported cash and short-term debt as hypothetical opening June 18 balances. From June 18 onward, assume daily net operating cash use of USD3,000, including both June 18 and the applicable deadline. The short debt stays outstanding at SOFR plus 300 basis points, simple ACT/360. At each calendar day's 9 a.m. New York reset, use the latest SOFR fixing already published; an effective-date fixing becomes available on its next SOFR publication day, and the latest available rate carries over nonpublication days. Interest and principal are due on the applicable deadline whether or not the offering closes. Assume the prior amendment's deadline had remained in force and no new funds were received. Calculate cash immediately after paying the short debt on that prior deadline. Say what the cash left on that earlier date shows about covering the burn and the loan, given the branch's assumptions.

**Output to adjudicate (turn 2):** `prior_close_cash_usd`: Store cash after the prior-date hypothetical debt repayment in USD as a finite number, rounded to 4 decimals from unrounded inputs.

**Candidate A:** `23971.1072`

**Candidate B:** `23970.4279`

## Q29

**Data sources:** fdic_bankfind, treasury_yield_curve

**Turn 1.** Match banks present at both 2021-12-31 and 2024-12-31 with at least USD 100 million of assets and positive deposits and equity at the first date. Report bank count, both dates' 10-year par yields and their basis-point change. Rank banks by starting securities/assets share, certificate ascending on ties. Form five groups using linearly interpolated 20th, 40th, 60th and 80th percentiles of ordinal ranks; each upper cutpoint belongs to the lower group. Report median starting share in the lowest and highest groups and the highest group's minimum share, using linear interpolation. Interpret the matched-panel boundary and the rate move's conditional price effect on a fixed-rate portfolio bought earlier.

**Output to adjudicate (turn 1):** `top_fifth_lower_bound_pct`: Store the securities share at which the highest fifth begins, in percent rounded to 4 decimals.

**Candidate A:** `36.7724`

**Candidate B:** `36.7628`

## Q30

**Data sources:** treasury_yield_curve, nyfed_reference_rates, cftc_cot

**Turn 1.** For futures-only CFTC 10-year Treasury note contract 043602 report dates from 2021-2025, inner-join exact same-date Treasury 2- and 10-year yields and SOFR. Define leveraged-money net share as long minus short divided by open interest. Report aligned count/boundaries and select the minimum 10-year-minus-2-year spread, ties earlier, with spread, SOFR and net share. Carry the weekly panel and interpret the relationship and material scope of the curve and positioning measures.

**Turn 2.** Through 2023-12-31, estimate by OLS the normalized long-run relation with 10-year yield dependent on an intercept, 2-year yield and SOFR. On its residual, estimate by OLS an intercept-bearing no-lag ADF regression of residual change on lagged residual; compute the coefficient standard error using residual sum of squares divided by observations minus two. Report count, long-run coefficients, residual n-1 standard deviation, ADF level coefficient and t statistic. Carry the normalized residual forward and interpret what this Engle-Granger-style diagnostic can and cannot license about cointegration.

**Turn 3.** Estimate the conditional rank-one error-correction equation for weekly 10-year-yield change on an intercept, lagged equilibrium error, contemporaneous 2-year-yield change and contemporaneous SOFR change. Report count, coefficients, R-squared and maximum absolute unscaled OLS normal-equation moment. Carry coefficients forward and interpret adjustment sign and contemporaneous-information limits.

**Output to adjudicate (turn 3):** `ecm_observation_count`: Store the ECM regression observation count as an integer.

**Candidate A:** `260`

**Candidate B:** `155`

## Q31

**Data sources:** philadelphia_fed_rtdsm, treasury_yield_curve, cftc_cot

**Turn 1.** Create a retrospective revision screen for annualized real GDP growth from 2024-Q1 through 2025-Q2. Retain only quarters with first, second, third and current estimates, rank them by the absolute current-minus-first revision, and break a tie by the earlier quarter. Report the retained-quarter count, selected quarter, its first and current rates, and the selected absolute revision minus the runner-up absolute revision. Briefly explain how the selected retrospective revision magnitude should be interpreted.

**Output to adjudicate (turn 1):** `largest_revision_quarter`: Store the selected quarter as a YYYY-Qn string.

**Candidate A:** `"2025-Q2"`

**Candidate B:** `"2025:Q2"`

## Q32

**Data sources:** treasury_auctions, treasury_yield_curve

**Turn 1.** Within calendar-2025 Treasury auction records, restrict the population to original-issue ten-year fixed-rate nominal notes: exclude reopenings, inflation-indexed securities and floating-rate securities, require a reported issue date, and require positive bid-to-cover, coupon and price. Rank the eligible auctions by bid-to-cover descending, breaking an exact tie by earlier auction date and then CUSIP ascending. Report the eligible count and the selected CUSIP, auction date, issue date, maturity date, bid-to-cover ratio, annual coupon percent and auction price per 100 face. Carry the selected note and issue date forward, and interpret the material limits of using this selection rule as an auction-demand comparison.

**Turn 2.** On that note's issue date, construct a semiannual analytical discount curve from the Treasury par yields at 6 months, 1, 2, 3, 5, 7, 10, 20 and 30 years. Use a sequential par-instrument bootstrap and linearly interpolate par yields in maturity for the missing half-year tenors. Report the curve date, the reported 10- and 30-year par yields, the bootstrapped 10- and 30-year discount factors, the corresponding semiannual-compounded spot yields, and the maximum absolute repricing residual per 100 face across synthetic par instruments at every half-year through 30 years. Carry the complete bootstrapped curve forward and explain the financial distinction between the reported par yields and the derived spot curve.

**Turn 3.** Using the curve carried forward, price the selected note per 100 face on its issue date with twenty regular half-year cash-flow periods and no accrued interest. Report the model price. Then rebootstrap after parallel one-basis-point upward and downward shocks to all nine quoted par nodes and report the two shocked prices, the standard symmetric effective duration and the standard symmetric effective convexity. Interpret the price reconciliation, the roles of duration and convexity, and the reconciliation's material inferential limits.

**Turn 4.** Complete a two-key-rate hedge for 100 face of that note. Define synthetic five- and ten-year par securities on the base analytical curve, keeping their base coupons fixed thereafter. For the target and both hedges, measure five- and ten-year par-node key-rate DV01 by increasing only the corresponding quoted par node by one basis point, re-bootstrapping the curve, and taking the signed price loss under that upward bump. Solve the two hedge face amounts that set both combined key-rate DV01s to zero, with positive face denoting a long position and negative face a short. Report the target's two key-rate DV01s, both hedge face amounts and the maximum absolute post-hedge residual. Finally use the next available Treasury curve observation after the issue date, hold all coupons and cash-flow grids fixed, and report that date, the unhedged target and hedged portfolio changes as next value minus base value per 100 target face. Interpret what the hedge test establishes and its material coverage limits.

**Output to adjudicate (turn 4):** `five_year_hedge_face_per100_target`: Store the face amount of the synthetic five-year hedge held against every 100 face of the target note, positive for long and negative for short, rounded to 6 decimals. This is a face amount, not the ratio of one hedge unit to one target unit.

**Candidate A:** `0.317865`

**Candidate B:** `0.003179`

## Q33

**Data sources:** fed_z1_vintages, ken_french_daily_factors

**Turn 1.** Work in the newest release of the Financial Accounts only. Corporate equities on the nonfinancial corporate business balance sheet appear under quarterly series codes sharing the digits 103064103, one beginning LM, one FA, one FU and one FR. Report how many quarter-over-quarter changes the instrument supports, its market value at the final quarter end, the net issuance the sector transacted summed across those quarters, and how many of those quarters carry a negative net issuance. Explain what a negative net issuance says about what the sector was doing.

**Turn 2.** For each of those quarters the change in market value separates into what the sector transacted and the revaluation of what it already held. Report how many quarters satisfy that separation to within one million dollars, the largest absolute residual across them, how many carry a transacted amount and a market-value change of opposite sign, and how many carry a revaluation larger in absolute size than the transacted amount. Explain what the sign disagreement means for reading a balance-sheet change as sector activity.

**Turn 3.** Bring in the daily research factor file. For each of those quarters compound the value-weighted market return including the risk-free rate across that quarter's trading days. Express each quarter's revaluation as a percent of the market value the quarter opened with and set the two series side by side. Report how many quarters match, the correlation across them, the median absolute difference in percentage points, and how many agree in sign. Explain what the closeness establishes and what it leaves open.

**Output to adjudicate (turn 3):** `matched_quarter_count`: Store how many quarters match a compounded market return as an integer.

**Candidate A:** `29`

**Candidate B:** `41`

## Q34

**Data sources:** treasury_marketable_securities, treasury_dts_public_debt

**Turn 1.** The monthly statement of the public debt lists every marketable Treasury security outstanding at each month end, and for the inflation-protected ones it reports separately how much of the principal has accrued with the price index. Working from the security-level rows and leaving the file's published class totals aside, report how many month ends the file covers and, at the last of them, how many inflation-protected rows it carries, how many distinct securities those rows describe, and the outstanding balance summed across those rows in billions of dollars. Explain why the row count and the security count differ.

**Output to adjudicate (turn 1):** `tips_issue_row_count`: Store how many security-level inflation-protected rows the final month end carries as an integer.

**Candidate A:** `135`

**Candidate B:** `134`

## Q35

**Data sources:** fema_nfip, noaa_storm_events

**Turn 1.** The federal flood insurance claim file records one settled claim per row, each row able to carry a code for what caused the damage. Working from the whole file, report how many claim rows it holds and how many of those carry a recorded cause of damage. Then report how many distinct causes those coded rows describe among them, and how many claim rows carry the single most frequent cause. Explain what the coding does and does not settle about why these properties flooded.

**Output to adjudicate (turn 1):** `distinct_cause_code_count`: Store the number of distinct causes of damage those coded rows describe as an integer.

**Candidate A:** `19`

**Candidate B:** `12`

## Q36

**Data sources:** eia_bulk_wti_daily, ofr_market_stress

**Turn 1.** Analyze a one-barrel WTI inventory's positive daily dollar price declines on dates from 2010-01-01 through 2024-12-31 that have both an EIA daily spot observation and a numeric OFR Financial Stress Index. Each decline uses the immediately preceding EIA quote in the complete history, with source quotes treated as exact cents. Define the marked loss as that positive decline when the same-day OFR index is strictly positive, and zero otherwise; retain all common dates in its distribution and chronology. Use the lower inverse empirical 95th percentile as threshold and strict exceedances. Fit a zero-location generalized Pareto distribution to the excesses by the first two unbiased sample L-moments, with positive shape denoting a heavy unbounded tail. Report the common count, exceedance count, threshold, shape and scale. Carry the excess law and chronological marked-loss sample forward and discuss the financial scope of this tail fit.

**Output to adjudicate (turn 1):** `threshold_exceedance_count`: Store the initial strict exceedance count as an integer.

**Candidate A:** `187`

**Candidate B:** `189`

## Q37

**Data sources:** fed_z1_vintages, ken_french_daily_factors

**Turn 1.** Work in the newest release of the Financial Accounts only. Corporate equities on the nonfinancial corporate business balance sheet appear under quarterly series codes sharing the digits 103064103, one beginning LM, one FA, one FU and one FR. Report how many quarter-over-quarter changes the instrument supports, its market value at the final quarter end, the net issuance the sector transacted summed across those quarters, and how many of those quarters carry a negative net issuance. Explain what a negative net issuance says about what the sector was doing.

**Output to adjudicate (turn 1):** `summed_net_issuance_usd_millions`: Store the summed net issuance across those quarters in USD millions, rounded to 4 decimals.

**Candidate A:** `-4246044.0`

**Candidate B:** `-1050755.0`

## Q38

**Data sources:** sec_nmfp_series_filings, sec_nmfp_class_yields, treasury_yield_curve

**Turn 1.** Using the SEC N-MFP share-class yield table joined to the series filings on the accession number, take every row whose yield date is 2025-06-30 and that carries a series identifier. Report how many class rows there are and how many distinct series they belong to. Then report how many of those series report more than one distinct share class, and among those, how many report a single identical seven-day gross yield across every class. Finally report the median number of distinct classes per series across all series at that date. Briefly explain what identical within-series gross yields do and do not establish about the class rows.

**Turn 2.** Before measuring anything, apply a reporting-quality screen to the same 2025-06-30 rows. Report how many rows carry a seven-day net yield of exactly zero alongside a positive gross yield, and how many of those belong to a share class whose reported class net assets are zero. Separately, across the entire yield table and not only this date, report how many rows carry a net yield strictly above the gross yield. Then build the screened population by keeping only 2025-06-30 rows with positive class net assets, a positive gross yield, and a net yield that is above zero and not above the gross yield, and report how many rows and how many distinct series survive. Briefly discuss what this screen can and cannot establish about the observations it excludes.

**Turn 3.** On the screened population, define each class's wedge as its gross yield minus its net yield, expressed in basis points. Report the median wedge and the mean wedge across classes. Then report the wedge weighted by each class's net assets. Report the largest wedge borne by any single class. Finally, among series that still have more than one screened class, identify the series whose cheapest and dearest classes are furthest apart, and report its name and that spread in basis points. Interpret the difference between the equal-class and asset-weighted averages using only measures available in the declared sources.

**Output to adjudicate (turn 3):** `widest_series_name`: Store the name of the series whose classes are furthest apart as text.

**Candidate A:** `"Wells Fargo Money Market Fund"`

**Candidate B:** `"Allspring Money Market Fund"`

## Q39

**Data sources:** sec_bdc, nyfed_reference_rates

**Turn 1.** Build a retrospective hypothetical loan book from Chicago Atlantic BDC, Inc.'s December 31, 2024 annual 10-K investment schedule filed March 31, 2025. Keep individually identified term loans, delayed-draw term loans, incremental term loans, senior secured notes and Series A senior notes with positive reported principal and fair value, an explicit nonnegative PIK rate and reported all-in rate, and maturity after June 30, 2025. Carry unrounded amounts across turns and round only reported outputs. Report the position count, opening principal and opening fair value in USD millions. Explain what comes from the filing and what will be hypothetical in the financing exercise.

**Turn 2.** For this frozen book, impose a new January 1–March 31, 2025 contract: each loan's contract coupon is the December 31, 2024 published SOFR plus 5 percentage points. Within that coupon, the loan's explicitly reported PIK percentage points of principal are capitalized at quarter end only, and the remainder of the coupon is paid in cash. Use 90 ACT/360 interest days and opening principal throughout the quarter. Opening cash is zero. Opening debt is 64.5% of opening fair value, with financing interest at the same SOFR plus 2.5 percentage points, paid March 31; new debt drawn that day carries no first-quarter interest. Report cash loan interest, capitalized PIK and financing interest in USD millions. Explain which part of the quarter's coupon adds to loan principal and which part is cash the fund can actually spend.

**Turn 3.** At March 31 pay a cash distribution of 80% of the quarter's total contractual loan interest, counting the capitalized PIK portion together with the cash-paid portion. Eligible collateral is only the opening fair value, advanced at 65%; newly capitalized PIK has zero eligible value this quarter. Fund the distribution and paid financing interest first with cash coupon receipts, then draw up to unused borrowing capacity, then inject the minimum external equity. Assume no fees and no other flows. Report the paid distribution, new end-quarter borrowing and equity injection in USD millions. Explain how the distribution and the financing interest compare with the quarter's cash coupon receipts, and what covers any gap.

**Turn 4.** Immediately after that payment, reduce by 15% the opening fair values of those positions whose reported PIK rate is strictly positive; all other opening marks stay fixed. The new PIK principal remains ineligible. Apply the same 65% advance rate and repay any borrowing-base excess immediately with additional external equity; the earlier distribution and financing payments remain executed. Report the revised borrowing base and additional equity cure in USD millions. Explain how the revised base compares with borrowing already drawn, and what has to happen when the two differ once the distribution has gone out.

**Turn 5.** Plan April 1–June 30 from the cured state, with no maturities or mark changes. Loan coupons reset to March 31, 2025 SOFR plus 5 percentage points; each loan retains its reported PIK percentage points. Use beginning-of-quarter capitalized principal, 91 ACT/360 interest days, and quarter-end PIK posting. Financing costs the same SOFR plus 2.5 percentage points on debt outstanding throughout this quarter. A fresh USD 3 million equity budget may be injected April 1 and applied to debt repayment or held as cash; any unused borrowing capacity can be drawn only June 30. With no other flows, choose the allocation that maximizes a nonnegative June 30 cash distribution while leaving zero cash and debt no greater than the post-shock borrowing base. Report the April 1 debt repayment, maximum distribution and ending loan principal in USD millions. Explain how applying the April 1 budget to debt compares with holding it as cash, and when any freed capacity can be redrawn.

**Output to adjudicate (turn 5):** `q2_early_debt_repayment_mn`: Store the optimal April 1 equity-funded debt repayment in USD millions, rounded to 4 decimals.

**Candidate A:** `0.0`

**Candidate B:** `3.0`

## Q40

**Data sources:** sec_financial_statements, sec_insider

**Turn 1.** Build a hypothetical IAS 33 EPS exercise for calendar 2025. In Allison Transmission's original 2024 10-K, accession 0000950170-25-019379, annual net income is profit attributable wholly to ordinary shareholders and basic weighted-average shares are the opening ordinary share base. Separately take Charter Communications (CIK 1091667) original Form 4 transactions dated in 2024 with code A, title Stock Options, underlying Class A Common Stock, and positive numeric exercise price and underlying shares; each distinct accession and derivative-transaction row is a template, keeping separate grants to separate owners. Report the template row count and opening basic EPS. Carry full precision between turns.

**Turn 2.** Use those templates to create fully vested, cash-exercise USD options outstanding throughout 2025, each covering 500 times its reported underlying shares at its reported USD strike. The hypothetical annual average share price is 125% of the unweighted median template strike. There are no unrendered services, additional exercise proceeds, tax effects or other share-based payment adjustments. Report incremental option shares in millions and option-only diluted EPS in USD per share. Explain the relationship between option coverage and the EPS denominator.

**Turn 3.** Add three hypothetical voluntary convertible debts outstanding all of 2025: A, USD 4 billion face at 5% annual interest, converting into 30 million shares; B, USD 4 billion at 8% into 30 million; C, USD 2 billion at 12% into 15 million. Interest is expensed at the stated rate with a 25% tax benefit; there are no discounts, fees, other conversion effects or conversions, and the opening profit, all from continuing operations, already includes full-year interest on all three. Apply IAS 33 to the options and debts together. Report included convertible IDs in inclusion order, diluted EPS and diluted weighted-average shares. Explain how the instrument selection relates to diluted EPS.

**Turn 4.** Replace the profitability scenario with continuing-operations loss equal to 20% of the original annual net-income template and discontinued-operations profit equal to 40% of that template. Both are after tax and attributable to ordinary shareholders; the continuing result includes full-year interest on all three debts. Keep the ordinary share base and all tool terms unchanged; no actual conversion or exercise occurs. Report the included potential-instrument set, continuing-operations diluted EPS and total diluted EPS in USD per share. Explain how the two profit components relate to instrument selection.

**Output to adjudicate (turn 4):** `loss_total_diluted_eps`: Store total diluted EPS in the loss scenario in USD per share, rounded to 4 decimals.

**Candidate A:** `1.6805`

**Candidate B:** `1.6282`

## Q41

**Data sources:** treasury_marketable_securities, treasury_auctions, nyfed_reference_rates

**Turn 1.** Prepare a retrospective hypothetical issuer refinancing exercise for 2025-01-01 through 2025-03-31. In the 2024-12-31 Monthly Statement of the Public Debt marketable-securities table, select only Bills Maturity Value CUSIP detail maturing inside that window, not totals. A CUSIP may have several issue-date lines: use its one nonnull outstanding-principal detail amount once, then allocate it to that CUSIP's maturity date; scale each reported USD-million amount by 0.000001 to create the hypothetical issuer's USD-million old-debt book. In the Treasury auction table, use the one Bill, 4-Week record for each issue date 2025-01-07, 2025-01-28, 2025-02-18 and 2025-03-04; its actual issue date, maturity date and price per 100 are candidate templates. The issuer is not the US Treasury, and the auction results are known only because this is a retrospective counterfactual. Open 2025-01-01 with USD0.3m cash and zero borrowing. A base bridge line can draw to cover each dated cash deficit and immediately sweeps any surplus to repay debt. Each day, take the New York Fed SOFR with the latest Effective Date on or before that calendar day and add 300 basis points; apply that rate retrospectively to outstanding debt at the start of the day and accrue simple ACT/360. Pay accumulated interest at whichever comes first of the next old-debt maturity, new-note maturity, candidate issue settlement, or the 2025-03-31 terminal event. New draws start interest the following day. At an event, pay accrued bridge interest, old-debt principal and any maturing new-note face in that order, clearing cash after each stage, before receiving any newly issued note proceeds; then sweep the proceeds to bridge debt. Before choosing any new notes, report the total scaled old principal due, the last old maturity date, and the base bridge principal required immediately before the first candidate issue's proceeds arrive. Explain how MSPD issue-date detail and settlement order affect those quantities.

**Turn 2.** Now let the hypothetical issuer choose 0 through 20 whole USD0.1m face lots at each of the four candidate issue dates. A lot brings the observed auction price-per-100 times its face only on the actual issue settlement date and owes its full face on the recorded maturity date. Its outstanding live new-note face may not exceed USD2.5m after any issue; this separate policy cap excludes old obligations and bridge debt. The base bridge principal may not exceed USD2.7m at any intermediate clearing stage, including before same-day issue proceeds; emergency credit is unavailable in this original plan. Follow turn 1's daily SOFR+300bp accrual and event-order cash sweep. Minimize through 2025-03-31 the aggregate new-note issue discounts plus base bridge interest, using full unrounded inputs and states. At March 31, keep any still-outstanding note face and bridge debt as terminal liabilities at par. An exact cost tie selects the lexicographically lower four lot counts. Report the four selected issue lot counts, full-window financing cost and highest base bridge principal immediately before any candidate issue settlement. Explain how the chosen lot sizes trade the discount paid at issue against the bridge the issuer must carry into each settlement date.

**Turn 3.** Treat the selected 2025-01-07 and 2025-01-28 lot counts from turn 2 as issued and irrevocable. Execute the dated old principal, new-note maturity, issue cash, bridge interest and sweeps through the end of all 2025-01-28 settlements using the unrounded plan. The later two turn-2 lot counts are still proposals. Report cash and base bridge principal immediately after the January 28 issue proceeds and cash sweep, in USD millions. Explain the obligations remaining from the two executed issues after the January 28 settlement.

**Output to adjudicate (turn 3):** `post_jan_28_cash_mn`: Store unrestricted cash after all 2025-01-28 ordered settlements and cash sweep, USD millions, rounded to 6 decimals.

**Candidate A:** `0.674931`

**Candidate B:** `0.0`

## Q42

**Data sources:** finra_short_volume, sec_ftd

**Turn 1.** FINRA's daily short-volume file is dated by trading day; the SEC fails-to-deliver file is dated by settlement date. Find the 2025 trading day with the largest total volume in the short-volume file, all reported shares rather than short sales alone, summed over symbols and breaking a tie by the earlier day. Report that day, the settlement date of its trades, and the shares failed on that settlement date summed over all securities. Explain how you matched the two dates.

**Turn 2.** On that settlement date, which symbol's failed shares rose most from the previous settlement date in the file, summing rows per symbol, reading a symbol absent on a date as zero and breaking a tie by symbol ascending? Report the symbol and its rise. Explain what the file's figure for a date measures.

**Turn 3.** Repeat the first turn for January through April 2024: report the busiest trading day, the settlement date of its trades, and the shares failed on that settlement date. Explain how you matched the dates.

**Turn 4.** Report the settlement dates of trades executed on 2024-05-24, on 2025-01-08 and on 2025-10-10. Explain what decided each one.

**Output to adjudicate (turn 4):** `settlement_for_2024_05_24`: Store the settlement date of the first listed day's trades as an ISO YYYY-MM-DD string.

**Candidate A:** `"2024-05-29"`

**Candidate B:** `"2024-05-28"`

## Q43

**Data sources:** sec_financial_statements, treasury_yield_curve

**Turn 1.** Construct a hypothetical IFRS 9 financial liability whose principal equals ACI Worldwide's reported noncurrent long-term debt in its original 2024 annual filing, accession 0000935036-25-000006 (principal only). The liability runs five years with a 5% annual coupon paid in exact half-year periods and bullet repayment. It is issued at the value of its cash flows on the 2024-12-31 Treasury yields for 1, 2, 3, 6 months and 1, 2, 3, 5, 7, 10, 20, 30 years, read as zero rates, semiannually compounded, linearly interpolated with flat ends, less a cash issuance fee of 1% of principal. Report the initial net carrying amount and the original semiannual effective interest rate.

**Turn 2.** Execute the first four scheduled original coupons. There are no rate resets, modifications or additional fees during these two years. Immediately after the fourth coupon, report the liability carrying amount and effective-interest expense for that fourth half-year period. Explain the relation between the coupon and the accounting interest charge.

**Turn 3.** At that date execute proposal A: keep the original principal, pay a 4% annual coupon, and extend the remaining term to eight half-years, with a lender fee of 0.4% of original principal and a third-party legal fee of 0.6%. Run the quantitative 10% test at the original effective rate; it alone is decisive. Follow the IASB's October 2017 clarification on modifications without derecognition; modification fees adjust a continuing liability and amortize over its modified term. Report A's absolute percentage change in the 10% test, the modification gain before fees, the fee-adjusted carrying amount and the semiannual fee-amortizing effective rate. Explain the role of the two fees and the accounting consequence.

**Output to adjudicate (turn 3):** `A_adjusted_carrying_mn`: Store post-A carrying amount after both modification fees in USD millions, rounded to 4 decimals, calculated from unrounded inputs and not from other reported rounded values.

**Candidate A:** `879.4861`

**Candidate B:** `861.6932`

## Q44

**Data sources:** fdic_bankfind, nyfed_reference_rates

**Turn 1.** From 2024-12-31 FDIC BankFind reports, keep banks with USD 5–10 billion total assets inclusive, positive cash, deposits and book equity, securities exceeding 30% of assets, and cash plus securities no greater than assets. Select the lowest cash-to-deposits ratio, breaking a tie by certificate ascending. Report its certificate, cash, securities and book equity/assets. Treat its reported cash and securities as available in a hypothetical deposit-run balance sheet; freeze all other assets and nondeposit liabilities at the opening accounting identity. Compare the bank's cash with its deposits and its securities, and say which of the two is the real source of liquidity in a run.

**Turn 2.** Model a first-period withdrawal of 35% of that bank's opening deposits. Securities may be sold in cumulative book-value tranches: first 25% at 97% of book, next 25% at 94%, remainder at 80%. A repo facility lends at most USD 1,000 million total. Each repo draw pledges exactly the securities book value needed at an 80% advance rate, with no excess pledge; pledged securities cannot also be sold. Execute the selected security sales and repo draw at period start; cash earns no interest, and the withdrawal occurs at period end. Repo interest for the full 90 days is simple ACT/360 at New York Fed SOFR effective 2024-12-31 plus 150 basis points, paid at period end. Choose sale and repo amounts to minimize recognized sale loss plus interest, while ending cash is at least 2% of the deposits remaining after that period's withdrawal and period-end book equity/assets is at least 8.5%. Report securities sold at book, repo principal and ending book equity/assets. Explain why the repo raises cash without reducing what the bank owes, while selling into the deepest tranche raises cash only by taking a loss against book equity.

**Turn 3.** As a comparison against the frozen opening balance sheet, find the largest first-period deposit withdrawal feasible with no new equity under the same sale prices, 90-day repo terms, ending cash reserve and book-equity/assets floor. This frontier does not replace the first-period trades already executed. Report the withdrawal amount, its share of opening deposits, and which cash-reserve, collateral and book-capital conditions bind. Interpret the role of funding choices and constraints in this capacity.

**Output to adjudicate (turn 3):** `frontier_bindings`: Store exactly one token from CASH_ONLY, CASH_COLLATERAL, CASH_CAPITAL, CASH_COLLATERAL_CAPITAL.

**Candidate A:** `"CASH_COLLATERAL"`

**Candidate B:** `"CASH_COLLATERAL_CAPITAL"`

## Q45

**Data sources:** eia_bulk_wti, ofr_market_stress

**Turn 1.** Across 2000-01-01 through 2025-12-31, restrict the analysis to dates on which both the reported EIA daily Cushing WTI spot series and the OFR Financial Stress Index have observations. Among every window of five consecutive joint observations, select the most negative ending-minus-starting WTI price-level change, breaking a tie by the earlier ending date. Report the joint-observation count, window dates, starting and ending prices, and the magnitude advantage over the runner-up decline (the second-largest five-observation decline, whether or not its window overlaps the selected one). Explain the financial measure and source represented by the selected price path.

**Turn 2.** For that selected window, report the second, third and fourth joint-observation dates and WTI prices, then report the window trough date and price, with an earlier date breaking a trough tie. Explain how this price path should be summarized and what a percentage-return description would and would not establish.

**Turn 3.** Carry the same start and end dates into the OFR table. Report starting and ending FSI, and ending-minus-starting changes in Credit, Equity valuation, Safe assets, Funding and Volatility contributions. Identify the category with the largest positive contribution change, breaking a tie alphabetically, and report that change. Interpret the category changes relative to the headline FSI change over the window.

**Output to adjudicate (turn 3):** `largest_positive_contribution_change_category`: Store the market category with the largest contribution increase.

**Candidate A:** `"credit"`

**Candidate B:** `"Credit"`

## Q46

**Data sources:** sec_nport, treasury_yield_curve

**Turn 1.** Use original SEC N-PORT-P reports dated 2025-03-31 and filed through 2025-06-30. Retain each series' latest filing, breaking filing-date ties by accession descending, and its complete USD five-tenor interest-rate-risk row at that exact accession. Require net assets of at least USD 100 million, positive cash not elsewhere reported that is strictly below net assets, and reported assets less liabilities within USD 1 of net assets. The five DV01 tenors run 3-month, 1-year, 5-year, 10-year then 30-year throughout. Subtract absolute signed-net DV01 magnitude from gross absolute five-tenor DV01 magnitude, scale that per-basis-point gap to a 50-basis-point move by multiplying it by 50, and express the result as a percentage of net assets. Select the series with the largest such measure, breaking ties by series identifier ascending. Report the selected series identifier, accession, net assets, cash not elsewhere reported and ranking measure. Separately, from complete Treasury par yields at those same five tenors on 2020-01-02 through 2024-12-31, form consecutive available-date changes in basis points and their sample covariance using the n-minus-one denominator. Report the change count and carry the selected balance sheet, signed DV01 vector and covariance forward. Explain from the five signed tenor sensitivities how a fund whose net DV01 nearly cancels can still carry substantial curve-shape exposure.

**Turn 2.** For the carried fund and Treasury covariance, find the minimum-Mahalanobis-distance five-tenor yield shock whose signed first-order USD DV01 loss equals 2% of the report-date net assets. Express every shock component in basis points; in this scenario a positive reported DV01 means a gain for a one-basis-point yield decline. There is no tenor bound yet. Report the minimum distance and the five-year and ten-year shock components. Interpret the shape of the resulting stress relative to a parallel move.

**Turn 3.** Keep the same fund, covariance and 2%-of-original-NAV loss target, but restrict each tenor's shock separately to the closed range from negative to positive 50 basis points. Resolve the minimum-distance stress within that box. Report its distance, ten-year shock component and the first attained tenor bound in the stated tenor order, or no bound if all are interior. This comparison neither executes a rate move nor alters the fund holdings. Say which tenor limit binds, and explain what that bound does to the distance required for the same target loss.

**Turn 4.** Now execute a hypothetical redemption equal to 20% of the original report-date net assets, at the reported balance-sheet amounts, with liabilities unchanged. Spend the reported cash first; sell the remaining required amount of noncash assets at book value pro rata across all noncash assets. Every USD DV01 bucket retains the same fraction as noncash asset book value; cash carries no DV01. Reassess the unboxed minimum-distance stress for a first-order loss of 2% of the new net assets, keeping the historical Treasury covariance. Report noncash assets sold, the fraction of the USD DV01 vector retained and the new minimum distance. Interpret the exposure relative to the remaining net assets and the scenario's sale assumption.

**Turn 5.** From the executed post-redemption state, use the original unboxed minimum distance from turn 2 as a radius and consider every five-tenor shock within it. Determine the new common-equity cash contribution that brings the worst first-order USD-rate loss to exactly 2% of net assets measured after that contribution. The contribution changes cash and net assets but not the carried DV01 vector or liabilities. Report the worst loss before the contribution as a share of post-redemption net assets and the required contribution. Explain what the post-redemption exposure implies for the original loss limit, and state that this is a first-order DV01 scenario, not a probability of loss.

**Output to adjudicate (turn 5):** `required_new_equity_mn`: Store the minimum fresh common-equity cash injection in USD millions, rounded to 4 decimals.

**Candidate A:** `16.9632`

**Candidate B:** `16.9628`

## Q47

**Data sources:** hmda, fhfa_hpi, nyfed_reference_rates

**Turn 1.** Establish the acquisition pool from 2024 HMDA Florida conventional first-lien originations for owner-occupied, one-unit home purchases. Retain numeric loan amounts from USD100,000 through USD750,000, annual incomes from USD40,000 through USD250,000, interest rates from 3 through 10 percent, and property values from USD100,000 through USD1,500,000, all inclusive. Retain identical public rows as separate exposures. Treat each retained loan as a hypothetical new 360-month, level-payment, fully amortizing loan at its reported nominal annual rate convertible monthly, with end-month principal-and-interest payments. Report the pool count, initial principal and aggregate monthly payment. Carry its individual exposures and contracts forward. Assess the scope of using these reported originations to construct the acquisition pool. Use unrounded inputs and state throughout.

**Turn 2.** Set the acquisition underwriting case. As a stipulated collateral overlay, index each property value by the Florida traditional purchase-only quarterly FHFA NSA index growth from 2023Q4 to 2024Q4, then apply a 20-percent haircut. Reduce each reported income by 20 percent. Every loan makes its first 24 scheduled payments; immediately afterwards it defaults exactly when annual scheduled principal-and-interest exceeds 35 percent of stressed income and its remaining balance exceeds stressed collateral. A default settles for 92 percent of stressed collateral, capped at that remaining balance; a nondefault loan settles for its remaining balance. There are no other prepayments, defaults or cash flows. Report the pool scheduled balance after payment 24 and its terminal credit loss. Carry this underwriting case and cash-flow state forward, and interpret the loss mechanism.

**Turn 3.** Prepare a separate FHA comparison pool using the same initial eligibility and hypothetical loan-contract conventions, except for FHA insurance in place of conventional loan type. Scale all its exposures proportionally to the acquisition pool’s initial principal. Apply the acquisition underwriting case to this comparison. Report its eligible loan count before scaling, equal-notional monthly payment and terminal credit loss. Compare the two pools once sized to the same starting principal, and say what such a gap can and cannot establish about insurance.

**Turn 4.** For the FHA comparison only, evaluate a severe alternative with 30-percent income reduction and a 30-percent collateral haircut after the same index overlay; all other comparison conventions remain. Report its equal-notional terminal credit loss and the fraction of original FHA principal assigned to default. Explain how the fraction of principal going bad differs from how much is ultimately lost per dollar once collateral is sold.

**Output to adjudicate (turn 4):** `fha_severe_loss_million`: Store equal-notional FHA severe terminal loss in USD millions, rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.

**Candidate A:** `6512.3518`

**Candidate B:** `20259.8883`

## Q48

**Data sources:** sec_13f_filings, sec_13f_holdings, sec_companyfacts

**Turn 1.** Form 13F filings carry a cover page whose value total is stated separately from the individual holding rows. Use Form 13F-HR submissions, including combination reports, but exclude notices and every amendment, and keep filings that report both a positive cover-page value total and a positive holding-row count. For the quarters ending 2022-09-30 and 2022-12-31 report the number of qualifying filings in each. For each quarter compute every filing's cover-page value total divided by its holding-row count and report the median of that quantity, then report the later quarter's median divided by the earlier one and, separately, the later quarter's summed cover-page totals divided by the earlier quarter's. Take every median and percentile asked for here with linear interpolation. Explain what the size of these two ratios says about whether the change between the quarters is an economic one.

**Output to adjudicate (turn 1):** `pre_break_filing_count`: Store the number of qualifying filings for the earlier period as an integer. Count submitted filings, not distinct filers: a manager that submits two qualifying 13F-HR reports for one period contributes two.

**Candidate A:** `3177`

**Candidate B:** `6745`

## Q49

**Data sources:** sec_companyfacts, sec_submissions

**Turn 1.** Use each CompanyFacts issuer's latest original 10-K accepted by year-end 2025 for a fiscal year ending in 2024; break acceptance ties by greatest accession. Require a common 350–380-day annual window ending on its report date, positive operating cash flow, and reported nonnegative PP&E purchases, common-stock repurchases and total dividends. Count all eligible issuers before any shortfall filter. Rank the entire cohort by signed distribution shortfall relative to post-capex operating cash flow, descending; break ties by CIK ascending. Report that count, leader's CIK, accession, acceptance timestamp and shortfall. Explain what the leader's figure says about how its payouts line up with the cash its own operations left after equipment spending, and what it leaves open.

**Turn 2.** For that filing, carry its annual flows and matching opening and closing cash-and-equivalent balances. Report closing cash, net investing inflow excluding PP&E purchases, net financing inflow excluding dividends and common-stock repurchases, and the signed cash-balance change unexplained by the three reported cash-flow totals. Interpret the financing contribution and the unexplained amount. Retain unrounded inputs throughout this case.

**Output to adjudicate (turn 2):** `other_investing_inflow_bn`: Store net investing inflow excluding PP&E purchases in USD billions as a finite number, rounded to 4 decimals from unrounded inputs.

**Candidate A:** `4.588`

**Candidate B:** `4.368`

## Q50

**Data sources:** sec_notes, sec_notes_dimensions, nyfed_reference_rates

**Turn 1.** Use Ares Capital's SEC filing-note investment facts at September 30, 2025 in accession 0001287750-25-000046. Match reported investment-level principal, fair value and paid-in-kind interest rate by the same investment dimension; retain rows with all three finite and positive. Treat reported PIK rates as annual decimal fractions of principal. Report the retained position count, total fair value and annual PIK-interest run-rate on current principal. Interpret what this static run-rate does and does not measure.

**Turn 2.** Using ARCC's original June 2025 10-Q accession 0001287750-25-000038 and September 2025 10-Q accession 0001287750-25-000046, isolate third-quarter 2025 reported PIK interest income from their year-to-date filing-note totals. From the September filing's third-quarter issuer-affiliation breakdown, report PIK interest attributed to unaffiliated issuers. Also report after-tax net investment income for that quarter. Compare the reported quarter's accrual with the prior turn's position run-rate, and interpret the unaffiliated component.

**Output to adjudicate (turn 2):** `q3_after_tax_nii_m`: Store Q3 2025 reported after-tax net investment income in USD millions, rounded to 4 decimals from its filing fact.

**Candidate A:** `338.0`

**Candidate B:** `349.0`

## Q51

**Data sources:** bis_credit_gap, bis_debt_service, bis_effective_exchange_rates

**Turn 1.** Build a common-economy retrospective screen using 2025-Q4 BIS all-lender private-nonfinancial-sector actual credit-to-GDP and actual-minus-trend credit gap, reported household, non-financial-corporate and private-nonfinancial-sector debt-service ratios, and broad nominal and real effective-exchange-rate indexes for 2024-12 and 2025-12. Retain economies with actual credit-to-GDP of at least 100%, reported private-sector DSR of at least 15%, and a negative percentage change in the real EER. Rank the retained economies by nominal EER percentage change from most negative upward, breaking a tie by display economy name ascending. Report the common count, retained count, selected economy and the runner-up change minus the selected change. Explain which hurdles an economy had to clear before it could top this ordering, and how far that standing carries outside the filtered group.

**Turn 2.** For the economy selected above, report its 2025-Q4 actual credit-to-GDP ratio and actual-minus-trend credit gap. Within the fixed common population from the prior turn, also report its descending ranks on actual credit and on the credit gap, breaking ties by display economy name ascending. Explain how this economy's two placements sit against each other, and what the second measure does and does not say about borrowing too much.

**Output to adjudicate (turn 2):** `selected_actual_credit_rank`: Store the selected economy's descending actual-credit rank in the common population as an integer.

**Candidate A:** `13`

**Candidate B:** `10`

## Q52

**Data sources:** sec_companyfacts, sec_midas_security_exchange

**Turn 1.** Company facts record cash-flow amounts with a start and an end date. Keep facts from annual reports on form 10-K whose reporting window runs between 350 and 380 days and whose end date falls in 2024, and where a company reports the same concept twice for such a window keep the value from the latest filing date. Build a population of companies reporting positive Net Income (Loss). Report how many companies that leaves and their combined Payments for Repurchase of Common Stock in USD billions, treating a company with no repurchase value as having repurchased nothing. Then report how many of those companies report Payments of Dividends, Common Stock, how many report the broader Payments of Dividends concept, and how many report both. Explain what the relationship between those three counts implies for anyone measuring dividends from this dataset, and why the duration filter matters for cash-flow facts.

**Turn 2.** Measure what the concept choice costs. Compute total dividends two ways: first using only Payments of Dividends, Common Stock and treating its absence as zero, then using the broader Payments of Dividends concept where present and falling back to Payments of Dividends, Common Stock where the broader concept is absent. Report both totals in USD billions. Report total repurchases divided by each of those two dividend totals. Then report how many companies pay dividends under the second rule while reporting nothing under the first. Explain how the two readings differ in what they would let a reader conclude about how American companies return cash, and what an absent concept can and cannot establish about nonpayment.

**Output to adjudicate (turn 2):** `misclassified_payer_count`: Store how many filers pay dividends under the broader-first rule but report nothing under the narrow tag, as an integer.

**Candidate A:** `115`

**Candidate B:** `114`

## Q53

**Data sources:** treasury_yield_curve, nyfed_reference_rates

**Turn 1.** Using dates observed in both the daily Treasury curve and the New York Fed SOFR series, select the date with the lowest reported 10-year-minus-2-year Treasury par-yield spread, breaking an exact tie by the earlier date. Report the eligible common-date count, selected date, its 2-year and 10-year yields, and SOFR. Carry the date forward and interpret the relationship between the two selected measures and its material limits.

**Turn 2.** Take the 252 Treasury curve observations ending on that selected date. At each date, linearly interpolate the reported 6-month, 1-, 2-, 3-, 5-, 7-, 10-, 20- and 30-year par yields onto the regular half-year grid through 30 years, sequentially bootstrap par instruments, and convert the resulting discount factors to continuously compounded zero yields. Estimate a Diebold-Li dynamic Nelson-Siegel panel with one shared positive decay parameter in [0.01, 5.00] per year and date-specific level, slope and curvature factors by global least squares, choosing the lower decay if an exact objective tie occurs. Report the window boundaries and count, fitted decay, panel zero-yield RMSE, the loading matrix's spectral (2-norm) condition number, and maximum bootstrap par-repricing residual. Carry the decay, loadings and factor history forward, and interpret the curve-fit and numerical diagnostics and the analytical status of the constructed curve.

**Turn 3.** Using the carried decay and selected-date factor vector, report the level, slope and curvature factors and the fitted continuous zero yields at 2, 10 and 30 years. From the same Nelson-Siegel curve, evaluate the instantaneous forward curve on the 0.25-year grid from 0.25 through 30 years and report its minimum maturity and rate, breaking an exact rate tie by the shorter maturity. Also report the maximum absolute unscaled OLS normal-equation moment max|X' residual| for the selected-date cross-section. Explain what the fitted factor values imply for the curve's shape at short, middle and long maturities, and what the forward low point does not prove.

**Turn 4.** Estimate a full intercept-bearing VAR(1) on the carried 252-date DNS factor history and make a one-step factor forecast for the next available Treasury curve date. Report that date, the three forecast factors and the transition matrix's spectral radius. Reconstruct the forecast curve using the carried decay and compare its half-year-grid RMSE against the observed bootstrapped zero curve with the RMSE from a factor random-walk forecast; report both RMSEs and whether the VAR is lower. Interpret this single forecast comparison and its material coverage limits.

**Output to adjudicate (turn 4):** `dns_forecast_beats_random_walk`: Store whether the DNS VAR forecast has lower curve RMSE than the factor random walk as a Boolean.

**Candidate A:** `false`

**Candidate B:** `true`

## Q54

**Data sources:** sec_bdc, nyfed_reference_rates

**Turn 1.** Use Ares Capital's investment schedule for September 30, 2025, accession 0001287750-25-000046 (CIK 1287750). For this hypothetical collateral facility, include investment rows whose reported principal, cost and fair value are all present, finite and strictly positive; keep each investment once. The contract counts the lesser of each row's principal and fair value as eligible value, advances 70% against that value to set the borrowing base, and starts with debt equal to 60% of the total eligible value rather than of the base. Report the eligible position count, opening borrowing base and opening debt. Carry each selected row's unrounded amounts and explain the borrowing-base measure.

**Turn 2.** Apply an immediate 20% fair-value markdown to every carried eligible position, leaving principal, reported adjusted cost basis and facility debt unchanged. Reassess the same facility covenant at the row level. Report the resulting excess of facility debt over the stressed borrowing base and assess the covenant position.

**Turn 3.** On November 3, 2025, at that stressed instant, sell the same percentage of every selected investment at 90% of its stressed fair value. Apply all cash proceeds immediately to repay facility debt; remove the same percentage of each position's principal, reported adjusted cost basis and stressed fair value from the collateral pool. There are no fees, taxes or position-size limits. Find the smallest sale percentage that restores debt to no more than the recomputed borrowing base. Report that percentage, signed realized sale proceeds less allocated reported adjusted cost basis, and the signed change in NAV caused by selling below the already stressed carrying fair value. Explain how selling assets changes both sides of the test you just solved, and compare the two loss figures you reported against what had already been marked.

**Turn 4.** Keep the loan principal and collateral left by that minimum immediate sale unchanged from November 3 through December 31, 2025, inclusive. Accrue simple daily interest on the remaining principal at the SOFR fixing effective on each calendar day, with no publication lag, plus 200 basis points, ACT/360; carry the latest fixing across nonpublication days. Unpaid accrued interest counts in covenant debt at each day-end but does not itself earn interest. At the December 31 valuation, after the last fixing is published, capitalize the accrual once. Report the ending excess debt over the unchanged borrowing base and assess whether the immediate cure lasts through the financing window.

**Output to adjudicate (turn 4):** `year_end_borrowing_deficiency_m`: Store the accrued year-end facility debt in excess of the unchanged post-sale borrowing base, in USD millions, rounded to 4 decimals from unrounded source inputs.

**Candidate A:** `88.6174`

**Candidate B:** `162.7201`

## Q55

**Data sources:** ofr_market_stress, ken_french_10_industry_portfolios, ken_french_daily_factors

**Turn 1.** Build one common-date panel from the daily OFR Financial Stress Index, the Ken French value-weighted 10-industry portfolios and the daily Fama-French factors. Among common dates from 2019-01-01 through 2025-12-31, select the maximum OFR FSI level, breaking an exact tie by earlier date. Let the training end be the common date exactly 20 observations before that peak, use 504 common observations ending there, and define the disjoint stress window from the common date immediately after that training end through the common date 40 observations after the peak, so the window straddles the peak. Report the peak date and FSI level, both training boundaries, both stress boundaries and the stress observation count. On the training window, rank all ten industries by annualized sample volatility using 252 days and report the five highest-volatility canonical Ken French industry tokens in descending order, breaking an exact tie alphabetically. Carry the windows and ordered five-industry cohort forward, and explain the information boundary created by this construction.

**Turn 2.** Using only the carried 504-date training window and the five retained industries, estimate the sample covariance matrix of daily decimal industry returns with the sample denominator. Report its minimum eigenvalue and spectral condition number. Separately run five Fama-French three-factor OLS regressions with an intercept, each industry's daily excess return as dependent variable, and Mkt-RF, SMB and HML in the source table's percentage units. Among the five fitted market betas, report the highest and lowest canonical industry tokens and values, breaking an exact beta tie alphabetically. Carry the covariance matrix and all fitted factor loadings forward, and interpret what the eigenvalue, condition number and fitted betas establish and do not establish.

**Turn 3.** From the carried sample covariance matrix, solve the fully invested, strictly positive, long-only equal-risk-contribution portfolio with equal variance-risk budgets across the five industries. Report each weight in the cohort order established in the first turn, the annualized training volatility using 252 days, and the maximum absolute deviation of a component variance-contribution share from 20 percent. Using the carried regression loadings, also report the portfolio's fitted Mkt-RF, SMB and HML betas. Carry the frozen weights forward and explain how equal risk contribution differs from equal capital weighting and what the factor exposures add to the risk-budget result.

**Turn 4.** Apply the frozen target weights to each daily-return vector in the carried 60-date stress window, treating both portfolios as daily rebalanced and performing no weight re-estimation. Report the tail observation count, compounded portfolio return, annualized realized volatility using 252 days, maximum drawdown from a wealth index initialized at one, and positive-loss 97.5-percent non-interpolated historical expected shortfall using the worst ceiling-tail count. For an equal-capital-weight portfolio on the same cohort and dates, report compounded return and expected shortfall under the same convention. On the ERC portfolio's own tail dates, calculate component expected-shortfall contributions from each weighted industry loss; report the largest contributor by its canonical industry token and its contribution, breaking an exact contribution tie alphabetically, and component-sum minus portfolio expected shortfall. Compare the two portfolios, explain what the component contributions represent and over which observations they are defined, and give the material limits of the tail estimate.

**Output to adjudicate (turn 4):** `erc_stress_maximum_drawdown_pct`: Store the ERC stress-window maximum drawdown as a positive percent, rounded to 4 decimals.

**Candidate A:** `38.6435`

**Candidate B:** `39.328`

## Q56

**Data sources:** ken_french_10_industry_portfolios, ofr_market_stress

**Turn 1.** Use Kenneth French's value-weighted HiTec, Enrgy, Utils and NoDur daily simple returns and the OFR Financial Stress Index on common complete dates from 2018-01-01 through 2022-12-31. Consider a daily-rebalanced portfolio with capital weights 40, 20, 20 and 20 percent in that order. Treat the joint observed return-and-stress scenarios as a uniform finite-population prior. Report the scenario count, prior daily portfolio volatility and lower-inverse empirical 95th-percentile OFR FSI level. Define the stress event as FSI strictly above that level. Carry the complete joint prior and event forward, and discuss the financial meaning and scope of the prior.

**Turn 2.** Apply Meucci's discrete Entropy Pooling to the carried prior, minimizing posterior-relative-to-prior Kullback-Leibler divergence on the unchanged scenario support. Give full confidence simultaneously to two equality views: the stress event has probability 20 percent, and the portfolio's expected daily simple return is 0.10 percent. Report the divergence using natural logs, the posterior inverse-Herfindahl effective scenario count and posterior daily portfolio volatility. Carry the entire posterior probability law forward and assess how its information content differs from the historical prior.

**Output to adjudicate (turn 2):** `entropy_divergence`: Store posterior-relative-to-prior Kullback-Leibler divergence using natural logs, rounded to 4 decimals. Compute from your own unrounded inputs and carried state, not other rounded outputs.

**Candidate A:** `0.1399`

**Candidate B:** `0.1425`

## Q57

**Data sources:** sec_notes, treasury_yield_curve

**Turn 1.** Using Apple's fiscal-2025 SEC Notes dataset facts with month-end-rounded period date 2025-09-30, build the undiscounted long-term-debt principal maturity ladder from the six separately tagged buckets: next twelve months, years two through five, and after year five. Report each bucket and their sum. Reconcile that sum to the reported consolidated gross long-term debt, its face amount before unamortized discount, issuance costs and hedge-accounting adjustments: report the number of all physical rows for that measure/date/unit, the number of physical rows carrying no dimensions, the number of distinct numeric values among those no-dimension rows, the consolidated gross-debt amount, and maturity sum minus gross debt. Explain what the counts you just tallied say about how the filing stores that measure, and what the ladder-versus-total agreement does and does not settle.

**Output to adjudicate (turn 1):** `gross_debt_fact_physical_row_count`: Store the physical matching fact-row count as an integer.

**Candidate A:** `4`

**Candidate B:** `1`

## Q58

**Data sources:** sec_financial_statements, sec_notes_dimensional

**Turn 1.** Select original full-year Form 10-Ks ending in calendar 2024, one per filer by latest report date, filing date, then accession. Report filer count. Among period-end balances, count filers with long-term debt total, its noncurrent portion, operating lease liability, goodwill and stockholders' equity; count those with positive equity and those whose debt total is below its noncurrent portion. Interpret that discrepancy and how you would resolve it.

**Turn 2.** Work with the filers carrying equity above zero. Report three medians of debt to equity, each taken with linear interpolation: on the noncurrent portion alone, on the whole long-term total, and on that whole total with the operating lease liability added to it. Then report those filers' operating lease liability summed. Explain what each of the three numerators leaves in or leaves out, and which of them a lender writing a covenant would want.

**Turn 3.** Stay with the same filers. Report their equity summed and their goodwill summed, how many of them are left with equity below zero once goodwill is taken out of it, and their summed long-term total plus lease liability divided by their summed equity after goodwill, summing across every one of those filers including the ones whose own equity after goodwill falls below zero. Explain what taking goodwill out of equity is meant to capture, and what a result below zero leaves of the ratio.

**Turn 4.** In the separate dimensional fact file, match business-segment goodwill balances to the exact filing used for each entity goodwill figure by filer, period end and accession. Use only primary presentation (index 0) for repeated members. Report matched-filer count and count whose segment sum is within USD 1 of entity goodwill. Among reconciled filers report median largest-segment share of goodwill, with linear interpolation, and count with one segment above half. Interpret the accession match, what an unreconciled segment sum permits you to conclude, and the concentration versus the prior entity-level ratio.

**Output to adjudicate (turn 4):** `single_segment_majority_count`: Store how many have one segment holding more than half of it, as an integer, rounded to 0 decimals from unrounded source rows.

**Candidate A:** `125`

**Candidate B:** `32`

## Q59

**Data sources:** sec_insider, sec_companyfacts

**Turn 1.** From SEC insider submissions select exact Form 4 filings (not amendments or Forms 3/5) with report period 2024-10-01 through 2024-12-31 and exactly one reporting owner. Report filing count, count with a Table I holding row, holding-row count, and row-weighted indirect-ownership shares for holding rows and all selected filings' Table I transaction rows. Interpret what the two row types capture and their indirect shares.

**Turn 2.** For a cross-sectional view, retain one filing per issuer and reporting owner: latest report period, ties by later filing date then higher accession. Carry those filings forward. Report issuer-owner pair count and holding-row count on retained filings. Explain how this retention rule affects interpreting a sum of reported positions across filings.

**Turn 3.** Using only the retained filings, total the holding-row shares for each issuer and select the issuer whose total is largest, breaking a tie by the lower CIK. Report that issuer's CIK, its holding-row share total, the indirect portion of that total, and how many reporting owners contribute to it. Explain what the number of contributing owners and the size of the indirect portion together indicate about the way this issuer's insiders hold their stock.

**Turn 4.** Report the largest single holding row among that issuer's retained filings. Take its cover-page common shares outstanding from XBRL facts, choosing the latest cover date on or before 2024-12-31, ties by later filing date then higher accession. Express T3's holding-row total as a percentage of that count. Explain how the numerator, denominator dates, row types and eligibility rules limit this percentage.

**Output to adjudicate (turn 4):** `selected_holdings_to_shares_outstanding_pct`: Store that holding-row total against shares outstanding, in percent, rounded to 4 decimals from unrounded source rows.

**Candidate A:** `3.3656`

**Candidate B:** `2.4615`

## Q60

**Data sources:** sec_nmfp_series_filings, treasury_yield_curve

**Turn 1.** For N-MFP series S000001111 reported June 30, 2025, select the latest filing date, breaking ties by greatest accession. Its reported cash plus portfolio-securities value defines a new hypothetical pool; reported WAM and WAL are exact target days for its noncash holdings. Eight divisible credit-free templates have rate-reset/final-maturity days (1,1), (7,7), (30,30), (90,90), (180,180), (1,90), (1,180), (1,365). Noncash opening market-value weights are nonnegative, sum to one and match both reported moments exactly. Cash is outside those moments. Report opening pool value and whether a compatible set exists. Retain full precision internally.

**Turn 2.** Within that compatible set, find the sharp minimum and maximum share of opening noncash value in templates with final maturity within seven calendar days inclusive. Interpret what the two bounds establish about the pool. For later turns, price the templates on the 2024-12-31 Treasury yields for 1, 2, 3, 6 months and 1, 2, 3, 5, 7, 10, 20, 30 years, read as zero rates compounded semiannually, interpolated linearly in years with flat ends, discounting on ACT/365 with no credit spread. The first five are zero-coupon claims worth USD 1 at opening; the last three have USD 1 principal and a locked first coupon set so that coupon plus principal at reset is worth USD 1, then reset to par, repaying principal only at final maturity.

**Turn 3.** At time zero, before any payment or reset, shift every zero yield up 100 basis points. Redeem 70% of opening pool units immediately at the opening USD 100 price. Use cash first, which earns nothing, then sell divisible holdings at stressed marks less execution spreads of 2, 5, 12, 35, 65, 18, 28 and 45 basis points of stressed value in template order, at least cost while funding exactly the payout; value retained holdings at stressed marks. Across every compatible book, report the maximum optimized execution cost and the minimum remaining-unit NAV. Explain how the unknown holdings relate to the guarantee, and read the execution-cost and remaining-NAV results together.

**Turn 4.** An additional investment mandate confirms total floating-note opening-value weight is at most 20%, with no 180-day fixed claim and no 365-day floating note. This restricts the same original compatible set; earlier redemption was an evaluation, not an executed portfolio change. Retain the shock and 70% redemption contract. Report the revised sharp upper seven-day final-maturity share, maximum optimized execution cost and whether all mandated books meet a cost budget of 9 basis points of opening pool value. Explain how the additional information changes the earlier assessment.

**Output to adjudicate (turn 4):** `capped_70pct_cost_mn`: Store maximum optimized execution cost under the additional mandate in USD millions, rounded to 4 decimals.

**Candidate A:** `0.1488`

**Candidate B:** `0.149`

## Q61

**Data sources:** ken_french_10_industry_portfolios, nyfed_reference_rates

**Turn 1.** Use all ten Kenneth French value-weighted industry daily simple returns on complete joint dates from 2024-01-01 through 2024-03-31 as equally likely states of a hypothetical one-day market. Each industry investment and the cash investment costs USD1 initially. Each industry investment returns its initial dollar together with that state’s simple-return gain or loss at the terminal date. The cash annual effective rate is the arithmetic mean of all published New York Fed SOFR fixings with effective dates in that quarter; one day is 1/252 of a year, and its terminal cash payoff is identical across states. Find the unrestricted Hansen–Jagannathan minimum-second-moment stochastic discount factor that prices every investment exactly under this empirical law. Report the scenario count, kernel mean, population standard deviation and minimum state value. Carry the complete empirical pricing model and kernel forward. Interpret its admissibility as a state-pricing model. Use unrounded inputs and state throughout.

**Turn 2.** In the carried market, obtain the Hansen–Jagannathan minimum-second-moment kernel subject also to nonnegativity in every state. Preserve the physical scenario probabilities, investment prices and cash payoff. Report its population standard deviation, largest state value and number of states with exactly zero kernel value. Carry the complete constrained kernel forward and interpret the economic effect of the constraint.

**Output to adjudicate (turn 2):** `nonnegative_kernel_sd`: Store dimensionless population standard deviation of the nonnegative kernel rounded to 4 decimals. Compute from your unrounded inputs and carried state, never from other rounded outputs.

**Candidate A:** `0.5074`

**Candidate B:** `0.5066`

## Q62

**Data sources:** treasury_yield_curve, dol_form5500_schedule_sb

**Turn 1.** Use Treasury's 2023-12-29 par yields at maturities one, two, three, five, seven, ten, twenty and thirty years as annual coupon rates for eight synthetic annual-pay par bonds. Each has unit face, integer-year coupon dates, price one and no accrued interest. Fit the Smith–Wilson discount curve exactly to these cash-flow instruments, with ultimate forward rate 3.5 percent annual effective and fixed convergence parameter alpha 0.1 per year. No automatic alpha calibration or credit adjustment applies. Report the discount factors at years 40 and 60, and the instantaneous forward rate at year 60 as a continuously compounded annual rate in percent. Carry the entire fitted curve, retaining unrounded state throughout. Discuss what the fit and long-horizon extrapolation represent financially for this synthetic instrument set.

**Output to adjudicate (turn 1):** `discount_factor_year40`: Store the year40 discount factor, rounded to 4 decimals. Compute from your unrounded source inputs and carried state, never from previously rounded outputs.

**Candidate A:** `0.2265`

**Candidate B:** `0.2258`

## Q63

**Data sources:** treasury_auctions, nyfed_reference_rates

**Turn 1.** Identify the original 26-week Treasury bill auctioned June 2, 2025 and issued June 5, excluding reopenings. Report its CUSIP, issue date, maturity date, published high accepted price per USD100 par and published high bank-discount rate. Explain what the auction price and quoted discount rate measure.

**Turn 2.** Carry that bill into a hypothetical purchase of USD10 million par at its published high price, with no fees. Hold to maturity and redeem at par. Report purchase cash and the simple annualized investment yield on cash paid, using actual issue-to-maturity days and a 365-day year. Explain what the gap between what is paid now and what comes back at maturity is pinned down by, under this hold-to-maturity assumption.

**Turn 3.** Finance 90% of that purchase cash with one hypothetical repo advance from issue date through the day before maturity; repay the advance and interest from maturity proceeds. Accrue simple interest for each calendar day from issue date through the day before maturity at the latest SOFR fixing already published by that day's 9 a.m. New York reset plus a constant 25 basis points, ACT/360; each effective-date fixing becomes available on the next SOFR publication day and carries across nonpublication days. Do not round daily interest. Report total repo interest and the constant repo spread over this realized SOFR path that would make the bill's redemption discount exactly cover financing interest at the same advance. Cash carry means the bill's redemption discount less the repo interest, with no charge for the unfinanced part. Compare the financing spread actually charged against the break-even one you solved for, and say what the buyer could have known in June.

**Output to adjudicate (turn 3):** `repo_interest_usd`: Store total baseline repo interest in USD as a finite number, rounded to 2 decimals from unrounded daily accrual.

**Candidate A:** `200185.0`

**Candidate B:** `200229.06`

## Q64

**Data sources:** hmda, ffiec_nic_institutions

**Turn 1.** The mortgage disclosure file reports debt-to-income in one column under more than one encoding: an exact integer inside one band of the scale, a named range outside it, and separate codes where the ratio was not collected. Take every record in the file. Report how many records it holds, how many distinct values that column takes, how many of those values can be placed on the ratio scale, and how many records carry one of the values that cannot. Explain what those last records are, and what the mixture of encodings means for summarising this column.

**Output to adjudicate (turn 1):** `ordered_class_count`: Store how many of those sit on the ratio scale, as an integer, rounded to 0 decimals from unrounded source rows.

**Candidate A:** `14`

**Candidate B:** `19`

## Q65

**Data sources:** sec_nport, treasury_yield_curve

**Turn 1.** Use original SEC N-PORT-P reports dated 2025-03-31 and filed through 2025-06-30. Retain each series' latest filing, breaking filing-date ties by accession descending, and its complete USD five-tenor interest-rate-risk row at that exact accession. Require net assets of at least USD 100 million, positive cash not elsewhere reported that is strictly below net assets, and reported assets less liabilities within USD 1 of net assets. The five DV01 tenors run 3-month, 1-year, 5-year, 10-year then 30-year throughout. Subtract absolute signed-net DV01 magnitude from gross absolute five-tenor DV01 magnitude, scale that per-basis-point gap to a 50-basis-point move by multiplying it by 50, and express the result as a percentage of net assets. Select the series with the largest such measure, breaking ties by series identifier ascending. Report the selected series identifier, accession, net assets, cash not elsewhere reported and ranking measure. Separately, from complete Treasury par yields at those same five tenors on 2020-01-02 through 2024-12-31, form consecutive available-date changes in basis points and their sample covariance using the n-minus-one denominator. Report the change count and carry the selected balance sheet, signed DV01 vector and covariance forward. Explain from the five signed tenor sensitivities how a fund whose net DV01 nearly cancels can still carry substantial curve-shape exposure.

**Turn 2.** For the carried fund and Treasury covariance, find the minimum-Mahalanobis-distance five-tenor yield shock whose signed first-order USD DV01 loss equals 2% of the report-date net assets. Express every shock component in basis points; in this scenario a positive reported DV01 means a gain for a one-basis-point yield decline. There is no tenor bound yet. Report the minimum distance and the five-year and ten-year shock components. Interpret the shape of the resulting stress relative to a parallel move.

**Turn 3.** Keep the same fund, covariance and 2%-of-original-NAV loss target, but restrict each tenor's shock separately to the closed range from negative to positive 50 basis points. Resolve the minimum-distance stress within that box. Report its distance, ten-year shock component and the first attained tenor bound in the stated tenor order, or no bound if all are interior. This comparison neither executes a rate move nor alters the fund holdings. Say which tenor limit binds, and explain what that bound does to the distance required for the same target loss.

**Output to adjudicate (turn 3):** `boxed_first_binding`: Store exactly one token from NONE, 3M_LOW, 3M_HIGH, 1Y_LOW, 1Y_HIGH, 5Y_LOW, 5Y_HIGH, 10Y_LOW, 10Y_HIGH, 30Y_LOW, 30Y_HIGH.

**Candidate A:** `"5Y_LOW"`

**Candidate B:** `"5Y_HIGH"`

## Q66

**Data sources:** treasury_auctions, treasury_yield_curve, nyfed_reference_rates

**Turn 1.** Build a hypothetical fixed-date delivery basket of Treasury CUSIPs 91282CBL4, 91282CJJ1 and 91282CKC4, in that order, using each original non-reopening auction's coupon, maturity and bid-to-cover. Value them on 2024-12-31 with that date's Treasury yields for 1, 2, 3, 6 months and 1, 2, 3, 5, 7, 10, 20, 30 years, read as zero rates compounded semiannually, interpolated linearly with flat ends. Coupons are regular semiannual flows anchored to maturity, keeping month-end when maturity is month-end, with no business-day adjustment. Discount on ACT/365 and accrue interest as actual elapsed days over actual coupon-period days. Report the lowest clean-price CUSIP, breaking ties by CUSIP ascending, and the second bond's dirty price.

**Turn 2.** Carry that basket to fixed delivery March 31, 2025. Finance each entire dirty purchase price at a bond-specific simple ACT/360 rate: December 31, 2024 SOFR plus 25 basis points times that bond's original auction bid-to-cover ratio. Reinvest any coupons paid after valuation through delivery inclusive at the same bond-specific rate until delivery. These assigned financing terms are hypothetical and remain fixed. Report delivery-date values of interim coupons for the first and third bonds, and the first bond's financed forward clean price, excluding delivery accrued interest. Explain how the cash-flow calendar affects delivery cost.

**Output to adjudicate (turn 2):** `bond1_carried_coupon_value`: Store delivery-date value of first bond interim coupons in USD per 100 face, rounded to 4 decimals, calculated from unrounded inputs and not from other reported rounded values.

**Candidate A:** `0.566`

**Candidate B:** `0.5702`

## Q67

**Data sources:** sec_financial_statements, treasury_yield_curve

**Turn 1.** Construct a hypothetical IFRS 9 financial liability whose principal equals ACI Worldwide's reported noncurrent long-term debt in its original 2024 annual filing, accession 0000935036-25-000006 (principal only). The liability runs five years with a 5% annual coupon paid in exact half-year periods and bullet repayment. It is issued at the value of its cash flows on the 2024-12-31 Treasury yields for 1, 2, 3, 6 months and 1, 2, 3, 5, 7, 10, 20, 30 years, read as zero rates, semiannually compounded, linearly interpolated with flat ends, less a cash issuance fee of 1% of principal. Report the initial net carrying amount and the original semiannual effective interest rate.

**Turn 2.** Execute the first four scheduled original coupons. There are no rate resets, modifications or additional fees during these two years. Immediately after the fourth coupon, report the liability carrying amount and effective-interest expense for that fourth half-year period. Explain the relation between the coupon and the accounting interest charge.

**Turn 3.** At that date execute proposal A: keep the original principal, pay a 4% annual coupon, and extend the remaining term to eight half-years, with a lender fee of 0.4% of original principal and a third-party legal fee of 0.6%. Run the quantitative 10% test at the original effective rate; it alone is decisive. Follow the IASB's October 2017 clarification on modifications without derecognition; modification fees adjust a continuing liability and amortize over its modified term. Report A's absolute percentage change in the 10% test, the modification gain before fees, the fee-adjusted carrying amount and the semiannual fee-amortizing effective rate. Explain the role of the two fees and the accounting consequence.

**Output to adjudicate (turn 3):** `A_ten_percent_test_pct`: Store absolute contractual present-value change for proposal A in percent, rounded to 4 decimals, calculated from unrounded inputs and not from other reported rounded values.

**Candidate A:** `2.8405`

**Candidate B:** `3.236`

## Q68

**Data sources:** sec_insider, ken_french_daily_factors

**Turn 1.** Select derivative transaction rows with reported row-level form type 4 and an ISO YYYY-MM-DD transaction date in 2024. Keep a row even if its submission document is an amendment, when its own form type is 4. Among code-M rows, report the count, counts with a conversion/exercise price and an expiration date, and median nonmissing conversion/exercise price in USD. Also count code-A rows in the same population. Use linear interpolation for medians. Interpret the SEC meanings and scope of M and A, and what the reported fields establish for later price analysis.

**Turn 2.** Apply a co-reporting match to the 2024 M rows: for each accession and transaction date, take the median numeric price across nonderivative S rows, join that price to every M row in the same accession and date, and retain matches with a positive conversion or exercise price and nonmissing underlying-share count. Report the matched M-row count, median conversion price, median co-reported sale price, median row-level sale-price-to-conversion-price multiple, and count whose sale price is below conversion price. Take every median and percentile asked for here with linear interpolation. Explain how the matching was constructed and what the resulting multiple does and does not establish about the underlying transactions.

**Turn 3.** For each matched M row, compute the signed co-reported price-gap proxy: sale-price summary less conversion price, times underlying shares on the derivative row. Report total in USD billions and median per row in USD. Rank issuers by summed proxy descending, issuer CIK ascending on ties; report the top two issuer names verbatim from their SEC submissions, the leader's total and M-row count, the runner-up's count, and issuer count. Use linear interpolation for the median. Interpret proxy amount versus row frequency and what the share count can establish.

**Output to adjudicate (turn 3):** `issuer_count`: Store how many issuers have at least one matched M row as an integer.

**Candidate A:** `167`

**Candidate B:** `166`

## Q69

**Data sources:** sec_notes, nyfed_reference_rates

**Turn 1.** Prepare a hypothetical January–June 2025 receivable wind-down cash budget for Mondelez International, Inc. using its original 2024-12-31 10-K filing in the SEC financial-statement-notes source, not a later amended or repeated filing fact. Restrict to that exact original accession's undimensioned USD instant note facts reported by the registrant itself, excluding any co-registrant rows, and identify the issuer CIK and accession. Freeze on-book net current receivables, unrestricted cash, cash including restricted cash and current restricted cash; reconcile unrestricted cash against cash including restricted cash and current restricted cash. Separately identify the disclosed principal amount of receivables already sold under its factoring program at that balance date. Those sold amounts are outside the owned opening receivable balance used below: assume the firm services and remits them at March end without a free-cash receipt. Treat all forward cash flows and financing terms below as hypothetical, not reported company agreements. Report issuer CIK, accession, owned receivables, unrestricted cash, cash including restricted cash, current restricted cash and already-sold outstanding face in USD millions. Explain which disclosed balances enter this budget and how the rights attached to them affect liquidity.

**Turn 2.** Use the turn-1 owned receivable balance as the only opening collection pool: 20%, 30%, 30% and 20% of that balance arrive at January, February, March and April month end respectively; nothing new is sold and there are no other business receipts. Hypothetical operating and investment cash payments at January–June month end are USD 2,300m, 1,800m, 500m, 200m, 100m and 100m. An initially empty revolver has a USD 600m limit. For each month take the New York Fed SOFR fixing with the latest Effective Date strictly before the first calendar day of that month, add 200 basis points, and charge simple interest for that month's actual calendar days on debt outstanding at the beginning of the month, on an ACT/360 basis. At month end put only still-owned collections, accrued interest and scheduled payments in one clearing balance; then draw just enough to avoid negative cash and use any positive cash to repay debt. Check cash and the credit limit at month end, with no separate intramonth draw or limit test. Newly drawn debt starts interest next month. For this no-factoring diagnostic only, let debt exceed the cap to identify the unfunded path. Report its peak required debt, aggregate six-month interest, and first month in which required debt exceeds USD 600m. Say in which month required debt first passes the limit, and explain why the month-by-month path, not the June cash position, is what reveals it.

**Output to adjudicate (turn 2):** `first_no_factor_breach_month`: Store the first month in which the no-factor path would need more than the credit cap, as a YYYY-MM string.

**Candidate A:** `"2025-02"`

**Candidate B:** `"2025-01"`

## Q70

**Data sources:** sec_bdc, nyfed_reference_rates

**Turn 1.** Use the business-development-company filing index. Report how many filers it covers, counted by distinct CIK rather than reported name, and how many close their fiscal year on a date other than 31 December. Then take every filing labelled fiscal year 2024, fiscal period Q3, and report how many there are, how many distinct calendar period ends they carry, and the number of days from the earliest of those period ends to the latest. Explain what the fiscal label is anchored to and what that implies for putting filings side by side on the strength of the label alone.

**Turn 2.** Retain one filing per filer under that label, keeping the one filed latest and breaking a tie by the higher accession, so that an amended filing supersedes the original it corrects. Report how many filers are retained, the calendar period end shared by the most of them as an ISO date, breaking a tie by the earlier period end, and how many retained filers carry some other period end. Explain what the retained set would represent to someone who treated the label as a single as-of date.

**Turn 3.** Join the retained filings to the investment schedule. A schedule row with no reported fair value contributes nothing, and rows carrying negative or zero fair values are kept as reported. Report how many retained filers carry any schedule row, the reported fair value across the retained filings, and the portion of it contributed by filers whose period end differs from the plurality date. Explain what the filer count and the value total each rest on, what an absent mark does to the total you report, and how much of that total was struck on some date other than the plurality one.

**Turn 4.** Now build the same cross-section by calendar period end instead, taking every filing whose period ends on the plurality date you identified, under the same one-per-filer retention and the same fair-value conventions. Report that cross-section's filer count and reported fair value. Then report overnight SOFR in effect at the earliest period end behind the fiscal label and at the plurality period end, taking in each case the published rate whose effective date is the latest on or before that day. Explain what changes between the two ways of assembling the cross-section, and what the two rates say about the conditions under which the label's marks were struck.

**Output to adjudicate (turn 4):** `calendar_fair_value_mn`: Store that cross-section's reported fair value in USD millions, rounded to 3 decimals from unrounded source rows.

**Candidate A:** `732648.089`

**Candidate B:** `1290044.225`

## Q71

**Data sources:** sec_bdc, sec_financial_statements

**Turn 1.** Use BlackRock Private Credit Fund's original 2025-09-30 10-Q filed by 2025-12-31. Report its accession, consolidated total assets and cash. Carry the filing forward.

**Turn 2.** In that filing's investment schedule, report the signed total of reported fair values. Separately count debt rows with positive principal, positive fair value and a stated future maturity, and report their principal. Count zero-principal rows with negative fair value and a stated future maturity, and report their signed value. How do these figures relate to balance-sheet cash?

**Output to adjudicate (turn 2):** `bdc_maturity_schedule_fair_m`: Store the signed sum of reported investment-schedule fair values in USD millions, rounded to 4 decimals from unrounded source values.

**Candidate A:** `2069.0926`

**Candidate B:** `3109.4862`

## Q72

**Data sources:** hmda, fed_z1_vintages

**Turn 1.** In 2024 HMDA, keep originated first-lien, owner-occupied, nonbusiness loans on site-built one-to-four-unit homes. Report their count and gross loan amount. Carry this cohort forward.

**Turn 2.** Within that cohort, report ordinary-refinance and cash-out-refinance counts and gross amounts separately. Interpret the cash-out amount.

**Output to adjudicate (turn 2):** `cashout_refinance_rows`: Store the cash-out-refinance reported-row count as an integer.

**Candidate A:** `455830`

**Candidate B:** `435013`

## Q73

**Data sources:** sec_nport_holding_details, sec_nport

**Turn 1.** Within the supplied selected holding-detail cohort, find the original N-PORT filing public by December 31, 2025 with positive net assets and the most debt-security rows with both C.9 status fields populated. Break a tie by lower accession. Give its accession, report date and qualifying row count.

**Output to adjudicate (turn 1):** `watchlist_status_rows`: Store number of selected-filing debt-security rows with both C.9 status fields populated as an integer.

**Candidate A:** `1177`

**Candidate B:** `3610`

## Q74

**Data sources:** sec_13f_holdings, sec_insider

**Turn 1.** In the 2024 Q4 governed 13F-HR aggregate, select the highest non-option common-share-value CUSIP among rows named ORACLE CORP with class COM (ties: CUSIP ascending). Report its CUSIP, filing-accession count, shares and value. What does the count represent? Carry the literal CUSIP forward.

**Output to adjudicate (turn 1):** `option_underlier_q4_accessions`: Store number of distinct 13F-HR information-table accessions aggregated for the CUSIP, as an integer, rounded to 0 decimals from unrounded inputs.

**Candidate A:** `4640`

**Candidate B:** `3283`

## Q75

**Data sources:** sec_insider, sec_midas_security_exchange

**Turn 1.** Use the supplied SEC insider filings with document type exactly Form 4 filed during calendar 2024. Eligible entries are non-derivative sales coded S and disposed D, dated during calendar 2024, with positive reported shares and price per share. Among filings with a reported trading symbol and at least one eligible entry whose filing date is seven or more calendar days after its transaction date, select the filing with the largest aggregate reported value of all its eligible sale entries; break ties by accession ascending. Report its accession, aggregate sale value and largest calendar-day interval. Explain what the summed dollar figure does and does not tell you about the seller's take, and what the day gap alone can support.

**Turn 2.** For that filing, take the eligible sale entries having the largest interval as a block. Search the supplied original Form 4 filings filed no later than 2024-12-31 for the earliest filing containing a matching block, breaking filing-date ties by accession ascending. A match requires the same issuer CIK, the same set of reporting-owner CIKs, and equality of transaction date, transaction code, acquisition/disposition indicator, security title, shares, price per share, post-transaction holdings and direct/indirect ownership for every entry in the block; retain repeated-entry multiplicity. Additional entries in the matching filing are permitted. Report the matching accession and filing date. Explain what finding another filing carrying the same entry details establishes about the reporting history of those facts, and what it leaves unresolved.

**Turn 3.** Use the selected issuer's reported trading symbol and the MIDAS Stock observations available through 2024-12-31. For each of the matched filing and the originally selected filing, define the analysis availability date as the first observed trading date strictly after its filing date. A pre-window consists of the ten observed dates immediately before availability; a post-window starts on availability and includes the next nine observed dates. Report both availability dates, each window's available observation count capped at ten, and for each filing whether the full ten-before/ten-after pair is available. Assess the suitability of these windows for a complete before/after comparison.

**Turn 4.** For each of those two post-windows, report mean daily on-exchange share volume and the window-wide hidden-volume percentage under the MIDAS measure's applicable trading-volume denominator. Interpret the comparison of market activity and its evidentiary scope. Use every observed date in the established post-window and retain input precision until the final outputs.

**Output to adjudicate (turn 4):** `matched_post_hidden_pct`: Store the matched filing's post-window hidden-volume percentage, rounded to 4 decimals.

**Candidate A:** `39.5121`

**Candidate B:** `47.6949`

## Q76

**Data sources:** eia_bulk_wti_daily, ofr_market_stress

**Turn 1.** Analyze a one-barrel WTI inventory's positive daily dollar price declines on dates from 2010-01-01 through 2024-12-31 that have both an EIA daily spot observation and a numeric OFR Financial Stress Index. Each decline uses the immediately preceding EIA quote in the complete history, with source quotes treated as exact cents. Define the marked loss as that positive decline when the same-day OFR index is strictly positive, and zero otherwise; retain all common dates in its distribution and chronology. Use the lower inverse empirical 95th percentile as threshold and strict exceedances. Fit a zero-location generalized Pareto distribution to the excesses by the first two unbiased sample L-moments, with positive shape denoting a heavy unbounded tail. Report the common count, exceedance count, threshold, shape and scale. Carry the excess law and chronological marked-loss sample forward and discuss the financial scope of this tail fit.

**Turn 2.** Using the carried peaks-over-threshold model with its empirical threshold-exceedance frequency, estimate the unconditional marked-loss 99-percent VaR and expected shortfall. Report both. Carry the full tail model and unrounded VaR forward and interpret the tail-risk estimates.

**Turn 3.** Raise the threshold to the carried 99-percent VaR. Use generalized Pareto threshold stability without refitting shape, and re-estimate the raised-threshold exceedance frequency from the carried complete marked-loss chronology. Estimate the runs extremal index with clusters separated by at least three consecutive nonexceeding common observations, counting both boundary clusters and without circular wrapping. Under the stationary Poisson-cluster approximation for a 252-observation block maximum, use that empirical frequency and extremal index. Report the raised exceedance count, extremal index and the block maximum's 95th and 99th percentiles. Carry the maximum law and both levels forward and interpret the dependence adjustment.

**Output to adjudicate (turn 3):** `extremal_index`: Store the extremal index as a dimensionless number, rounded to 4 decimals. Compute from its own unrounded source inputs or carried model state, not from other rounded outputs.

**Candidate A:** `0.8125`

**Candidate B:** `1.0`

## Q77

**Data sources:** sec_nmfp_flows, sec_nmfp_series_filings

**Turn 1.** The SEC daily shareholder flow file reports gross subscriptions and gross redemptions for each share class on each day, while net assets and fund category are reported once per series filing. Report how many rows the flow file holds and how many distinct share classes appear. Then join the flows to their series filing on the accession number. A series files once per reporting period and an amendment supersedes the filing it corrects, so where a series period carries more than one filing keep only the latest, and where an original and its amendment carry the same date keep the amendment. Sum the share classes of the retained filing on each flow date and keep observations whose reported net assets are positive. Report how many series-day observations result, how many distinct series they cover, and the median number of share classes per series-day.

**Turn 2.** Express each series-day's net flow, subscriptions minus redemptions, as a percentage of the series' reported period-end net assets. Report how many series-days carry net assets below one million dollars, and how many observations show a net flow whose absolute value exceeds 100 percent of assets. Then keep only series-days with net assets of at least one million dollars, report how many remain, and report the most negative net flow among them as a percent of assets. Explain the timing mismatch between a daily flow and the period-end asset denominator and what the remaining extreme implies for interpreting the ratio.

**Output to adjudicate (turn 2):** `small_asset_days`: Store how many series-days report net assets below one million dollars as an integer.

**Candidate A:** `783`

**Candidate B:** `492`

## Q78

**Data sources:** ffiec_call_reports, treasury_yield_curve

**Turn 1.** Build a hypothetical IFRS acquisition from the 2025-09-30 consolidated Call Reports of buyer RSSD 197478 and target RSSD 141556, treated as independent banks. Book equity is equity attributable to a bank's own shareholders, excluding inherited subsidiary NCI. The buyer pays 1.4 times the book equity of an 80% stake. Appraise the other 20%: 5% of its book equity at years 1 to 5 plus 1.4 times its book equity at year 5, discounted at the 2025-09-30 Treasury yields for 1, 2, 3, 6 months and 1, 2, 3, 5, 7, 10, 20, 30 years read as zero rates, semiannually compounded, linearly interpolated with flat ends. Report consideration and appraised NCI.

**Turn 2.** Execute the acquisition under IFRS 3 and IAS 32 with full-goodwill fair-value NCI. Inherited subsidiary NCI, consolidated assets less liabilities less book equity, stays separate at that amount, which is also its fair value, and nothing here changes it. All target assets are identifiable, with no prior goodwill; mark them down by 1% of original total assets, leaving liabilities unchanged and taxes zero. Pay half the consideration in cash and half in new ordinary shares; acquisition costs of 1% of consideration and issuance costs of 2% of issued equity are paid in cash. Report recognized goodwill and consolidated equity attributable to the parent. Explain the equity attribution in the acquisition.

**Output to adjudicate (turn 2):** `new_goodwill_mn`: Store newly recognized full goodwill in USD millions, rounded to 4 decimals, calculated from unrounded inputs and not from other reported rounded values.

**Candidate A:** `246.756`

**Candidate B:** `219.349`

## Q79

**Data sources:** bls_qcew, noaa_storm_events

**Turn 1.** Establish a hypothetical parametric payroll-protection book for Louisiana counties using 2019 BLS QCEW private-sector county NAICS-sector annual wages. Require positive, nonmissing wages for both manufacturing, NAICS31–33, and health care and social assistance, NAICS62. Select the five counties with the largest manufacturing wages, breaking ties by ascending county FIPS. The original book insures 25 percent of each selected county’s manufacturing annual payroll, distributed uniformly over 365 calendar days. Report total insured annual payroll and the largest-exposure county. Assess exposure concentration. Use unrounded state throughout.

**Turn 2.** For the original book, use NOAA County-geography Flood and Flash Flood records in those counties with source begin datetimes in 2021. A contract trigger interval runs from source begin through 48 hours after source end, clipped at 2022-01-01. Measure naive source-local elapsed hours without time-zone or daylight-saving adjustment. A NOAA episode is one occurrence; within a county-occurrence, overlapping trigger intervals cover time only once. Each county-occurrence has a 24-hour aggregate time deductible and a USD500,000 indemnity limit. Indemnity covers the insured payroll accruing during trigger time beyond that deductible. Aggregate county indemnities within each occurrence. The annual book limit is USD8 million, applied by earliest in-scope source begin datetime and then ascending episode identifier. Report total indemnity before the annual limit and the largest occurrence indemnity. Explain how the biggest single-occurrence figure is built up from the county pieces underneath it, and what the per-county ceiling does there.

**Turn 3.** Quote an independent health-care and social-assistance book for the same counties. Insure 25 percent of their corresponding annual payroll and use a 12-hour aggregate time deductible; retain all other trigger, limit and ordering terms. Report its insured annual payroll and total indemnity before the annual limit. Compare the two books on what they insure and what they pay, and explain which terms and locations drive any difference between those two outcomes.

**Turn 4.** Allocate that comparison book’s annual capacity across its occurrences. Report the episode in which the annual limit is first exhausted and the percentage of that occurrence’s pre-annual-limit indemnity that is paid. Explain what determines how much of that occurrence gets paid, working back through what earlier events had already drawn from the yearly pot.

**Turn 5.** Complete the annual-capacity allocation for the original committed manufacturing book. Report its exhaustion episode and the amount paid on that occurrence. Say what sets the point at which this book runs out of yearly room, and how that compares with the other book's timing.

**Turn 6.** For the original book’s actual payments under its annual limit, apply a per-occurrence reinsurance layer of USD750,000 excess of USD250,000, with no annual reinsurance cap or premium cash flow. Report total ceded claims and the episode with the largest retained payment, breaking a tie by ascending episode identifier. Explain which slices of each occurrence payment move to the reinsurer and which stay with the original writer under this layer's boundaries.

**Output to adjudicate (turn 6):** `largest_retained_episode`: Store largest original retained-payment episode identifier exactly as a source string.

**Candidate A:** `"156912"`

**Candidate B:** `"158892"`

## Q80

**Data sources:** ofr_market_stress, ken_french_10_industry_portfolios

**Turn 1.** Across calendar-2020 dates common to the daily OFR Financial Stress Index and the Ken French value-weighted 10-industry portfolios, find the largest increase in the OFR FSI between two consecutive common dates, resolving an exact tie by the earlier ending date. Report the common-date count, the window dates, and the starting and ending FSI index values. Carry the window forward and explain why the analysis is restricted to dates common to both sources.

**Turn 2.** Among the five OFR market-category contributions, credit, equity valuation, safe assets, funding and volatility, identify the largest contribution increase over that window, breaking an exact tie alphabetically, and report its name and increase in FSI index points. Explain how the five category figures combine into the overall index and its move, and how they stand next to the regional breakdown of the same measure.

**Turn 3.** On the ending date of that window, identify the worst- and best-return Ken French industry portfolios, breaking an exact return tie alphabetically by the official portfolio label. Report their labels and returns, the number of negative-return portfolios, and the equal-weighted arithmetic mean of all ten portfolio returns. Explain what this same-day dispersion does and does not establish about the stress move.

**Output to adjudicate (turn 3):** `worst_event_date_industry_label`: Store the worst-return official Ken French 10-industry label; casing and repeated whitespace are not significant.

**Candidate A:** `"Energy"`

**Candidate B:** `"Enrgy"`

## Q81

**Data sources:** treasury_auctions, philadelphia_fed_rtdsm

**Turn 1.** The Treasury auction records give a reference CPI for each inflation-protected security. Using those records, report the CPI-U index level, not seasonally adjusted, for July 2025 and the 12-month inflation rate to July 2025. Explain how you obtained the index level from the records.

**Turn 2.** The Philadelphia Fed real-time data set carries the CPI seasonally adjusted, as annualized monthly changes. Compound its twelve changes from August 2024 through July 2025 into a 12-month rate, converting each to a monthly factor by the twelfth root of one plus the rate, once from the most recent values and once from the first releases. Report both rates and how many of the twelve monthly figures changed after first release. Explain how they compare with the previous turn's rate and why inflation-protected securities are indexed as they are.

**Turn 3.** Return to the auction records. Report the index level for September 2025 and the 12-month inflation rate to September 2025 on the same basis as the first turn. Explain how you reached each month's level.

**Output to adjudicate (turn 3):** `september_2025_twelve_month_pct`: Store the 12-month inflation rate to September 2025 in percent, rounded to 2 decimals.

**Candidate A:** `3.01`

**Candidate B:** `3.0`

## Q82

**Data sources:** finra_short_volume, sec_ftd

**Turn 1.** FINRA's daily short-volume file is dated by trading day; the SEC fails-to-deliver file is dated by settlement date. Find the 2025 trading day with the largest total volume in the short-volume file, all reported shares rather than short sales alone, summed over symbols and breaking a tie by the earlier day. Report that day, the settlement date of its trades, and the shares failed on that settlement date summed over all securities. Explain how you matched the two dates.

**Turn 2.** On that settlement date, which symbol's failed shares rose most from the previous settlement date in the file, summing rows per symbol, reading a symbol absent on a date as zero and breaking a tie by symbol ascending? Report the symbol and its rise. Explain what the file's figure for a date measures.

**Turn 3.** Repeat the first turn for January through April 2024: report the busiest trading day, the settlement date of its trades, and the shares failed on that settlement date. Explain how you matched the dates.

**Output to adjudicate (turn 3):** `early_peak_settlement_date`: Store its trades' settlement date as an ISO YYYY-MM-DD string.

**Candidate A:** `"2024-01-17"`

**Candidate B:** `"2024-01-18"`

## Q83

**Data sources:** ken_french_10_industry_portfolios, ken_french_daily_factors, ofr_market_stress

**Turn 1.** Calibrate an original Fama–French three-factor model for the value-weighted HiTec industry portfolio on complete common daily observations from 2015-01-01 through 2019-12-31 with nonpositive OFR Financial Stress Index. Use intercept-bearing homoskedastic OLS on industry excess returns with Mkt-RF, SMB and HML. Report the calibration count, market loading and value loading. Retain the complete fitted model and interpret its conditional exposures. This is a retrospective study using the supplied frozen histories; use unrounded state throughout.

**Turn 2.** Commit an original factor-neutral program based on that fitted model. At every daily reset it invests its entire prior-day NAV in HiTec and adds zero-initial-value contracts delivering the three factor returns, with signed exposure multiples that neutralize the fitted loadings. There are no transaction costs, collateral flows or other cash flows. Report fitted arithmetic annual alpha and annualized regression residual standard error, using 252 observations per year and a residual-variance denominator of the calibration count minus four. Explain how the fitted excess return stands against the variability the regression leaves unexplained once the factor exposures are offset by the contracts.

**Turn 3.** Prepare an independent program based instead on a fit to strictly positive OFR FSI dates in the same calibration years, retaining the model specification and factor-contract conventions. Report its market loading and gross factor-contract exposure as a multiple of NAV. Explain what accounts for any difference in gross contract size between this program and the committed one, and whether the initial stake plays a part.

**Turn 4.** Evaluate that comparison program on dates with complete HiTec returns, three-factor data and OFR FSI observations in common from 2020-01-01 through 2024-12-31, with its fitted exposure multiples fixed and daily resets as specified. Start NAV at USD100 million immediately before the first observation. Report compounded total return and maximum drawdown as a positive loss percentage. For a drawdown tie use the earliest trough and then the earliest preceding peak; the initial NAV is eligible as a peak. Explain how the cumulative result over the window and the largest interim decline fit together, rather than describing either figure on its own.

**Output to adjudicate (turn 4):** `comparison_holdout_return_pct`: Store comparison compounded held-out total return in percent, rounded to 4 decimals. Compute from your own unrounded source inputs and carried state, never from other rounded outputs.

**Candidate A:** `35.0151`

**Candidate B:** `52.1001`

## Q84

**Data sources:** sec_financial_statements, sec_companyfacts

**Turn 1.** Select original full-year Form 10-Ks ending in calendar 2024, one per filer by latest report date, filing date, then accession. Attach each filer's similarly selected 2023 report only if its period end is 350–380 days earlier. Keep reports with 2024 operating, investing and financing cash-flow totals and cash/equivalents in both reports; count a joint filing once. Report the cohort count and how many have the three flows within USD 3 of the cash/equivalents change. What else can move cash between balance-sheet dates?

**Output to adjudicate (turn 1):** `statement_population_count`: Store the count of filers carrying all three totals and both cash balances as an integer.

**Candidate A:** `3642`

**Candidate B:** `3644`

## Q85

**Data sources:** sec_ftd, sec_financial_statements

**Turn 1.** Work with settlement failures recorded from 2024-07-01 through 2024-12-31 inclusive, keyed by trading symbol. A balance stands on a date whether or not the file reports a price for it. Report how many securities are recorded on at least half the window's settlement dates. Among them report the one with the largest average balance per settlement date in the window, breaking a tie by the lower symbol, together with that average and how far it stands above the runner-up. Explain what recurrence on half the window's dates selects for.

**Output to adjudicate (turn 1):** `largest_average_balance_shares`: Store its average balance in shares, rounded to 4 decimals.

**Candidate A:** `3946780.301`

**Candidate B:** `3226336.2778`

## Q86

**Data sources:** sec_companyfacts, sec_submissions

**Turn 1.** Use each CompanyFacts issuer's latest original 10-K accepted by year-end 2025 for a fiscal year ending in 2024; break acceptance ties by greatest accession. Require a common 350–380-day annual window ending on its report date, positive operating cash flow, and reported nonnegative PP&E purchases, common-stock repurchases and total dividends. Count all eligible issuers before any shortfall filter. Rank the entire cohort by signed distribution shortfall relative to post-capex operating cash flow, descending; break ties by CIK ascending. Report that count, leader's CIK, accession, acceptance timestamp and shortfall. Explain what the leader's figure says about how its payouts line up with the cash its own operations left after equipment spending, and what it leaves open.

**Turn 2.** For that filing, carry its annual flows and matching opening and closing cash-and-equivalent balances. Report closing cash, net investing inflow excluding PP&E purchases, net financing inflow excluding dividends and common-stock repurchases, and the signed cash-balance change unexplained by the three reported cash-flow totals. Interpret the financing contribution and the unexplained amount. Retain unrounded inputs throughout this case.

**Turn 3.** Project one hypothetical year from that closing cash balance. Operating cash flow falls 20 percent from the selected year's amount; PP&E purchases, total dividends and common-stock repurchases retain their selected-year amounts. Every other cash movement is zero. Report ending cash under the unchanged distribution plan and assess its cash-buffer consequence.

**Turn 4.** Require ending cash of at least half the scenario's opening cash. Keep the preceding assumptions, allowing only common-stock repurchases to fall, with no negative repurchases or increase above the original plan. Report the maximum retained repurchase amount. Separately, vary operating cash flow alone and report its minimum annual amount that would support the full original repurchase plan at the same cash floor. Explain what the retained buyback figure protects under this scenario's fixed commitments, and whether any of it reflects what the company has actually announced.

**Output to adjudicate (turn 4):** `maximum_retained_repurchases_bn`: Store the maximum repurchase amount permitted by the scenario in USD billions as a finite number, rounded to 4 decimals from unrounded inputs.

**Candidate A:** `0.435`

**Candidate B:** `0.765`
