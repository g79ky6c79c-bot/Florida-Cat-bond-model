import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import poisson, lognorm, genpareto  # 'gdp' n'existe pas, c'est 'genpareto'. Je le laisse ici au cas où vous voudriez l'utiliser plus tard.
from scipy.optimize import minimize
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# 1. Assumptions & Parameters
# =============================================================================

print("--- Initialisation des paramètres ---")

# Bond Structure
notional = 100_000_000  # USD
maturity = 3  # years
risk_free_rate = 0.02  # 2% per annum
attachment_point = 0.3  # 30% of notional (Trigger start)
exhaustion_point = 0.7  # 70% of notional (Trigger end)

# Simulation settings
n_sim = 10_000
np.random.seed(42)

# Parameters for Data Generation (Synthetic Florida Data)
num_years_history = 20
avg_hurricanes_per_year = 2
mean_loss_historical = 1.5e9
std_loss_historical = 1.2e9

# =============================================================================
# =============================================================================
# 2. Data Loading / Generation 
# =============================================================================

# Génération de données synthétiques
annual_losses = []
annual_counts = []  # On stocke le nombre d'événements ici

# Paramètres lognormaux
sigma_h = np.sqrt(np.log(1 + (std_loss_historical/mean_loss_historical)**2))
scale_h = mean_loss_historical * np.exp(-(sigma_h**2)/2)

for _ in range(num_years_history):
    num_hurricanes = poisson.rvs(avg_hurricanes_per_year)
    annual_counts.append(num_hurricanes) # On sauvegarde le compte
    
    yearly_loss = 0
    if num_hurricanes > 0:
        losses = lognorm.rvs(s=sigma_h, scale=scale_h, size=num_hurricanes)
        yearly_loss = np.sum(losses)
    annual_losses.append(yearly_loss)

loss_data = pd.DataFrame({
    "Year": range(2000, 2000 + num_years_history), 
    "Annual_Loss": annual_losses,
    "Event_Count": annual_counts # Ajout de la colonne des comptes
})

# =============================================================================
# 3. Catastrophe Risk Modeling 
# =============================================================================

# Frequency: Poisson process

lambda_est = loss_data["Event_Count"].mean() 

print(f"Modèle calibré - Fréquence (Lambda estimé): {lambda_est:.2f}")

# Severity: Lognormal fit sur les pertes non-nulles
losses_only = loss_data[loss_data["Annual_Loss"] > 0]["Annual_Loss"]
# On fit la lognormale sur les montants, ça c'est correct.
shape_lognorm, loc_lognorm, scale_lognorm = lognorm.fit(losses_only, floc=0)

# =============================================================================
# 4. Monte Carlo Simulation
# =============================================================================

def simulate_losses(n_sim, lambda_p, shape, loc, scale):
    simulated_losses = []
    for _ in range(n_sim):
        # 1. Determine number of events this year
        num_events = poisson.rvs(lambda_p)
        
        # 2. Determine severity of each event
        if num_events > 0:
            events_loss = lognorm.rvs(s=shape, loc=loc, scale=scale, size=num_events)
            total_year_loss = np.sum(events_loss)
        else:
            total_year_loss = 0
            
        simulated_losses.append(total_year_loss)
    return np.array(simulated_losses)

print("Lancement de la simulation Monte Carlo...")
simulated_losses = simulate_losses(n_sim, lambda_est, shape_lognorm, loc_lognorm, scale_lognorm)

# =============================================================================
# 5. Loss Distribution & Bond Payoff Logic
# =============================================================================

annual_loss_dist = pd.Series(simulated_losses)

# Seuils en dollars
attachment_loss = attachment_point * notional
exhaustion_loss = exhaustion_point * notional

# Calcul des probabilités
prob_attachment = np.mean(annual_loss_dist >= attachment_loss)
prob_exhaustion = np.mean(annual_loss_dist >= exhaustion_loss)

def calculate_bond_cashflows(simulated_losses_arr):
    # Vecteur de cashflows
    # Coupon + Principal à risque
    # Note: Simplification - on regarde la perte annuelle vs le trigger
    
    payoffs = []
    for loss in simulated_losses_arr:
        # Principal Recovery Factor
        if loss < attachment_loss:
            recovery = 1.0
        elif loss >= exhaustion_loss:
            recovery = 0.0
        else:
            # Linear erosion
            recovery = 1 - (loss - attachment_loss) / (exhaustion_loss - attachment_loss)
        
        payoffs.append(recovery)
    
    return np.array(payoffs)

recoveries = calculate_bond_cashflows(simulated_losses)
principal_losses = 1 - recoveries

# =============================================================================
# 6. Pricing & Valuation
# =============================================================================

expected_principal_loss = np.mean(principal_losses)
expected_loss_pct = expected_principal_loss # EL %

# Fair Spread Approximation
# Spread required to compensate for EL + Risk Premium
# Simple formula: Spread ~ EL + Risk_Margin
fair_spread = expected_loss_pct * 1.5 # Arbitrary 1.5x multiplier for risk aversion

# VaR & CVaR (95%)
var_95 = np.percentile(simulated_losses, 95)
cvar_95 = simulated_losses[simulated_losses >= var_95].mean()

# =============================================================================
# 7. Sensitivity Analysis (CORRIGÉ)
# =============================================================================

# On teste la sensibilité en multipliant la FRÉQUENCE (lambda), pas les pertes
multipliers = [0.8, 1.0, 1.2, 1.5]
sens_results = []

for m in multipliers:
    # On simule avec un lambda modifié
    sens_losses = simulate_losses(n_sim//5, lambda_est * m, shape_lognorm, loc_lognorm, scale_lognorm)
    
    # Calcul des métriques pour ce scénario
    p_att = np.mean(sens_losses >= attachment_loss)
    p_exh = np.mean(sens_losses >= exhaustion_loss)
    mean_l = np.mean(sens_losses)
    
    sens_results.append({
        "Freq_Multiplier": m,
        "Expected_Loss_Mean": mean_l,
        "Prob_Attachment": p_att,
        "Prob_Exhaustion": p_exh
    })

sensitivity_df = pd.DataFrame(sens_results)

# =============================================================================
# 8. Outputs & Visualization
# =============================================================================

print("\n=== RÉSULTATS DU MODÈLE CAT BOND ===")
print(f"Montant Nominal: ${notional:,.0f}")
print(f"Trigger (Attachment): ${attachment_loss:,.0f} (Pertes > {attachment_point:.0%})")
print(f"Trigger (Exhaustion): ${exhaustion_loss:,.0f} (Pertes > {exhaustion_point:.0%})")
print("-" * 30)
print(f"Expected Loss (EL): {expected_loss_pct:.2%}")
print(f"Prob. Attachment: {prob_attachment:.2%}")
print(f"Prob. Exhaustion: {prob_exhaustion:.2%}")
print(f"Fair Spread (approx): {fair_spread:.2%}") # Spread over risk-free
print(f"VaR (95%): ${var_95:,.0f}")
print("-" * 30)
print("Tableau de Sensibilité (Fréquence):")
print(sensitivity_df[["Freq_Multiplier", "Prob_Attachment", "Prob_Exhaustion"]])

# Graphiques
plt.figure(figsize=(14, 5))

# Plot 1: Distribution des Pertes
plt.subplot(1, 3, 1)
plt.hist(annual_loss_dist, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(attachment_loss, color='orange', linestyle='--', linewidth=2, label='Attachment')
plt.axvline(exhaustion_loss, color='red', linestyle='--', linewidth=2, label='Exhaustion')
plt.title('Distribution des Pertes Annuelles Simulées')
plt.xlabel('Perte (USD)')
plt.ylabel('Fréquence')
plt.legend()

# Plot 2: Structure du Payoff (Bond Recovery)
x_vals = np.linspace(0, exhaustion_loss * 1.5, 100)
y_vals = [1 if x < attachment_loss else (0 if x > exhaustion_loss else 1 - (x - attachment_loss)/(exhaustion_loss - attachment_loss)) for x in x_vals]
plt.subplot(1, 3, 2)
plt.plot(x_vals, y_vals, color='green', linewidth=3)
plt.title('Structure de Remboursement du Principal')
plt.xlabel('Perte Catastrophe (USD)')
plt.ylabel('Facteur de Remboursement (1.0 = 100%)')
plt.grid(True, alpha=0.3)

# Plot 3: Sensibilité
plt.subplot(1, 3, 3)
plt.plot(sensitivity_df["Freq_Multiplier"], sensitivity_df["Prob_Attachment"], marker='o', label='Prob. Attachment')
plt.plot(sensitivity_df["Freq_Multiplier"], sensitivity_df["Prob_Exhaustion"], marker='x', label='Prob. Exhaustion')
plt.title('Sensibilité à la Fréquence des Ouragans')
plt.xlabel('Multiplicateur de Fréquence')
plt.ylabel('Probabilité')
plt.legend()

plt.tight_layout()
plt.show()
