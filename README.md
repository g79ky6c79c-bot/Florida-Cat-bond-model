# Catastrophe Bond (Cat Bond) Pricing & Risk Modeling

This project implements a complete quantitative modeling framework for a Catastrophe Bond (Cat Bond) using Monte Carlo simulation, inspired by real-world insurance-linked securities (ILS) structures.

The objective is to model, price, and analyze the risk profile of a USD 100 million catastrophe bond exposed to hurricane risk, following a frequency–severity approach commonly used in actuarial science and reinsurance.

Project Overview

Instrument: Catastrophe Bond

Notional: USD 100,000,000

Maturity: 3 years

Risk Type: Natural catastrophe (hurricanes – Florida-style exposure)

Modeling Approach:

Poisson process for event frequency

Lognormal distribution for event severity

Monte Carlo simulation

Attachment & exhaustion triggers

Expected loss, fair spread, VaR, CVaR

Sensitivity analysis on catastrophe frequency

This project is suitable for:

Quantitative finance portfolios

Risk management / insurance analytics roles

CFA / Master-level academic projects

ILS & reinsurance modeling demonstrations

Modeling Framework
1. Bond Structure

Attachment Point: 30% of notional

Exhaustion Point: 70% of notional

Trigger Logic:

No principal loss below attachment

Linear principal erosion between attachment and exhaustion

Full principal loss above exhaustion

2. Catastrophe Risk Modeling
Frequency

Modeled using a Poisson process

Parameter λ calibrated as the historical average number of events per year

This corresponds to the maximum likelihood estimator for Poisson frequency

Severity

Event severity modeled using a Lognormal distribution

Parameters estimated from non-zero annual losses

Captures heavy-tailed loss behavior typical of catastrophe risk

3. Monte Carlo Simulation

10,000 simulated annual scenarios

For each scenario:

Number of catastrophe events is drawn

Loss severity for each event is simulated

Annual aggregate loss is computed

Bond principal recovery is determined

 Key Outputs & Metrics
Pricing & Risk Metrics

Expected Loss (EL)

Probability of Attachment

Probability of Exhaustion

Fair Spread (risk premium approximation)

Value at Risk (95%)

Conditional VaR (CVaR 95%)

Sensitivity Analysis

Stress-testing on event frequency (λ)

Impact on:

Attachment probability

Exhaustion probability

Expected losses

Visualizations

The model produces three key plots:

Distribution of Simulated Annual Losses

With attachment and exhaustion thresholds

Bond Principal Recovery Profile

Payoff structure as a function of catastrophe losses

Sensitivity to Event Frequency

Attachment & exhaustion probabilities vs frequency multiplier

Technologies & Libraries

Python

numpy

pandas

scipy

matplotlib

No external data sources are required — the model uses synthetic but realistic catastrophe data, making it fully reproducible.

 How to Run
python {name of the file}


The script will:

Calibrate the catastrophe model

Run Monte Carlo simulations

Print pricing and risk results

Display graphical outputs

Notes & Assumptions

Loss data is synthetically generated to mimic real hurricane exposure

Coupon pricing is simplified and focuses on principal-at-risk dynamics

Fair spread estimation includes a risk aversion multiplier (illustrative)

The model prioritizes clarity and financial intuition over excessive complexity

Possible Extensions

Real historical data (NOAA / EM-DAT / FEMA)

EVT (Generalized Pareto) severity modeling

Multi-year loss aggregation

Correlated multi-peril modeling

Investor vs issuer valuation comparison

Integration into a web-based dashboard

 # Author

Toussaint
Finance & Quantitative Risk Modeling
Focus: Asset Management, Insurance-Linked Securities, Market Risk
