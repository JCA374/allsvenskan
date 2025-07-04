Awesome. You're in a perfect spot to take advantage of **“Team Strength Extraction from Odds”** — especially now that you already have upcoming match odds and a working Poisson model. Let me break it down more deeply and give you a practical plan.

---

## 🧠 **What Is “Team Strength Extraction from Odds”?**

Bookmakers' odds already *encode* their belief about team strengths and expected goals.

If we can **reverse-engineer** those beliefs from the odds, we can:

* Estimate **attack/defense strength** directly from the market
* Use these strengths to feed a **Poisson simulation**
* Blend this with your historical/team-based model = true **Hybrid**

---

## 🎯 **What You Have:**

* Odds for upcoming games ✅
* Poisson model trained on historical goals ✅
* Probabilities from odds (using `remove_margin()`) ✅

---

## 🧩 **The Goal:**

Estimate team attack/defense strengths such that the **Poisson-predicted match probabilities match the market’s odds** as closely as possible.

---

## ⚙️ **How It Works (Simplified Version)**

Let’s say for a match:

* Market-implied probabilities:
  `P_home_win`, `P_draw`, `P_away_win` from odds

We want to find values of:

* `λ_home = base_attack_strength_home * base_defense_weakness_away * home_advantage`
* `λ_away = base_attack_strength_away * base_defense_weakness_home`

Then:

* Simulate Poisson distribution of goals with these λ’s
* Compute `P_home_win`, `P_draw`, `P_away_win` from that
* Minimize the difference between simulated and market probabilities

This is an **optimization problem.**

---

## 📦 **Implementation Plan**

### Step 1: Convert Odds → Probabilities

Use your existing `odds_converter.py`.

```python
p_home, p_draw, p_away = remove_margin(home_odds, draw_odds, away_odds)
```

---

### Step 2: Build Poisson Probability Calculator

You probably already have this, but it should:

```python
def match_probs_poisson(lambda_home, lambda_away, max_goals=6):
    probs = np.zeros((max_goals, max_goals))
    for i in range(max_goals):
        for j in range(max_goals):
            probs[i, j] = poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away)

    p_home = np.sum(np.tril(probs, -1))  # i > j
    p_draw = np.sum(np.diag(probs))      # i == j
    p_away = np.sum(np.triu(probs, 1))   # j > i
    return p_home, p_draw, p_away
```

---

### Step 3: Optimize Strengths for One Match

Use `scipy.optimize.minimize` to find λ\_home and λ\_away that match the bookmaker.

```python
from scipy.optimize import minimize

def objective(params, p_market):
    lambda_home, lambda_away = params
    p_model = match_probs_poisson(lambda_home, lambda_away)
    return np.sum((np.array(p_model) - np.array(p_market))**2)

res = minimize(objective, x0=[1.5, 1.2], args=([p_home, p_draw, p_away]))
```

---

### Step 4: Generalize to Many Matches

Now you can optimize **team strengths** instead of just λs directly:

```python
# Example variables to optimize:
# strength = {'Hammarby_attack': 1.2, 'Malmo_defense': 0.8, ...}
# λ_home = attack_home * defense_away * home_adv
```

This becomes a multi-match optimization problem — or you just do it **match-by-match**, store the λs, and average them per team over time.

---

## 💾 **Bonus: Save Strengths to DB**

Once you extract implied `λ_home`, `λ_away` from each match:

* Store them in your existing DB
* Over time, **track team form from market perspective**
* Compare to historical Poisson strengths → detect mispricing 📈

---

## 🔁 **Hybrid Usage**

Later in your `HybridPoissonOddsModel`:

```python
λ_poisson = team_strength_model(...)
λ_odds = extracted_from_market[match_id]

λ_final = weight * λ_poisson + (1 - weight) * λ_odds
```

Let me know if you want me to write a ready-to-use Python module for this (`odds_strength_extractor.py`) or integrate it into your model pipeline.
