# Fast Fashion Datasets — Streamlit Ready

Five clean CSVs, ready to `pd.read_csv()` in Streamlit. Total size ~10KB — loads instantly, no caching needed.

## File Map

| File | Rows | Best For |
|---|---|---|
| `us_prices.csv` | 148 | Line chart of US price categories over time |
| `bangladesh_wages.csv` | 36 | Wage freeze timeline; real wage line |
| `country_wages.csv` | 10 | Bar chart / map — living wage gap by country |
| `tshirt_breakdown.csv` | 6 | Waterfall / pie / stacked bar |
| `combined_divergence.csv` | 37 | Hero divergence chart (US + BD in one file) |

---

## Schemas

### `us_prices.csv` (long format — one row per Year × Category)
| Column | Type | Description |
|---|---|---|
| Year | int | 1990 – 2026 |
| Category | str | Apparel / Overall / Food / Housing |
| Raw_CPI | float | Original BLS index value |
| Index_2000 | float | Rebased to 2000 = 100 (main plotting column) |

**Streamlit tip:** Long format plays well with `st.line_chart` / `plotly.express`:
```python
import streamlit as st, pandas as pd, plotly.express as px
df = pd.read_csv('us_prices.csv')
fig = px.line(df, x='Year', y='Index_2000', color='Category')
st.plotly_chart(fig)
```

### `bangladesh_wages.csv`
| Column | Type | Description |
|---|---|---|
| Year | int | 1990 – 2025 |
| BD_CPI_2010base | float | Bangladesh CPI, 2010 = 100 |
| BDT_per_USD | float | Market exchange rate |
| RMG_Wage_BDT | float | Nominal monthly minimum wage (garment sector) |
| Real_Wage_2010BDT | float | Wage deflated to 2010 taka |
| Wage_USD_Market | float | Wage converted at market FX |
| Wage_USD_PPP | float | Wage in purchasing-power-adjusted USD |
| Is_Revision_Year | int | 1 if a wage revision happened this year, else 0 |

### `country_wages.csv`
| Column | Type | Description |
|---|---|---|
| Country | str | 10 major garment-exporting countries |
| Region | str | South Asia / Southeast Asia / East Asia / Middle East |
| Min_Wage_USD | int | Legal minimum wage, USD/month |
| Living_Wage_USD | int | Estimated living wage, USD/month |
| Gap_USD | int | Living − Minimum |
| Coverage_% | float | Min ÷ Living × 100 |
| Regime | str | Exploited / Moderate / Protected (categorical) |

### `tshirt_breakdown.csv`
| Column | Type | Description |
|---|---|---|
| Component | str | Cost stage of a $10 t-shirt |
| Cost_USD | float | Absolute dollar contribution |
| Category | str | Labor / Production / Logistics / Retail |
| Beneficiary_Country | str | Where that dollar goes |
| Share_% | float | Cost / 10 × 100 |
| Cumulative | float | Running total for waterfall plotting |

### `combined_divergence.csv` (the hero chart's fuel)
| Column | Type | Description |
|---|---|---|
| Year | int | 1990 – 2026 |
| Apparel_Idx, Food_Idx, Housing_Idx, Overall_Idx | float | US CPI categories, indexed 2000 = 100 |
| BD_Real_Wage_Idx | float | Bangladesh real wage, indexed 2000 = 100 |
| Wage_USD_Market | float | Same wage in USD |

Everything on one axis (2000 = 100), so plotting is trivial:
```python
df = pd.read_csv('combined_divergence.csv')
st.line_chart(df.set_index('Year')[['Apparel_Idx','Overall_Idx','BD_Real_Wage_Idx']])
```

---

## Data Sources
- **US CPI series** — US Bureau of Labor Statistics via FRED (series CPIAPPSL, CPIAUCSL, CPIUFDSL, CPIHOSSL)
- **Bangladesh CPI & FX** — World Bank Open Data (FP.CPI.TOTL, PA.NUS.FCRF)
- **Bangladesh RMG wages** — Bangladesh Minimum Wage Board public revisions
- **Living wage estimates** — Asia Floor Wage Alliance & Global Living Wage Coalition (illustrative)
- **T-shirt cost structure** — Clean Clothes Campaign & Fashion Revolution reports (illustrative)

## Deploy Checklist
1. Push these 5 CSVs + a `README.md` + your `app.py` to GitHub
2. Add `requirements.txt` with: `streamlit pandas plotly scikit-learn`
3. Connect the repo at share.streamlit.io — deployment takes 2 minutes
4. Files are small enough that no `@st.cache_data` is required, but adding it never hurts

## Suggested App Structure
- **Sidebar:** country selector, year range slider, view toggle
- **Tab 1 — The Divergence:** interactive plotly of combined_divergence
- **Tab 2 — Wage Freeze:** dual-axis with a slider highlighting each revision
- **Tab 3 — Country Comparison:** filterable bar of country_wages
- **Tab 4 — T-Shirt Calculator:** slider that scales tshirt_breakdown, showing what happens if worker wage doubles
- **Tab 5 — ML Playground:** let users pick k for K-means, watch cluster boundaries change
