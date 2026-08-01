"""
The Hidden Ledger of a $10 T-Shirt
Streamlit app — Fast Fashion: Wage vs Price Divergence

Deploy: push this + all 5 CSVs to GitHub, connect at share.streamlit.io
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# =============================================================
# PAGE CONFIG
# =============================================================
st.set_page_config(
    page_title="The Hidden Ledger | Fast Fashion Analytics",
    page_icon="👕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================
# CUSTOM STYLING
# =============================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #7f8c8d;
        font-style: italic;
        margin-top: 0;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #c0392b;
    }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; }
</style>
""", unsafe_allow_html=True)

# =============================================================
# DATA LOADING (cached)
# =============================================================
@st.cache_data
def load_data():
    us = pd.read_csv('us_prices.csv')
    bd = pd.read_csv('bangladesh_wages.csv')
    countries = pd.read_csv('country_wages.csv')
    tshirt = pd.read_csv('tshirt_breakdown.csv')
    combined = pd.read_csv('combined_divergence.csv')
    return us, bd, countries, tshirt, combined

try:
    us, bd, countries, tshirt, combined = load_data()
except FileNotFoundError as e:
    st.error(f"❌ Data file missing: {e}. Make sure the 5 CSVs are in the same directory as app.py.")
    st.stop()

# =============================================================
# HEADER
# =============================================================
st.markdown('<p class="main-header">The Hidden Ledger of a $10 T-Shirt</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">How fast fashion transfers cost from consumers to garment workers — a data story</p>', unsafe_allow_html=True)
st.markdown("---")

# =============================================================
# SIDEBAR
# =============================================================
with st.sidebar:
    st.markdown("### 🎛️ Filters")
    year_range = st.slider(
        "Year range",
        int(combined['Year'].min()),
        int(combined['Year'].max()),
        (2000, int(combined['Year'].max())),
    )
    st.markdown("---")
    st.markdown("### 📚 Data Sources")
    st.caption(
        "• **US CPI:** FRED / BLS  \n"
        "• **Bangladesh CPI & FX:** World Bank  \n"
        "• **Wage revisions:** Bangladesh Minimum Wage Board  \n"
        "• **Living wage:** Asia Floor Wage Alliance  \n"
        "• **T-shirt breakdown:** Clean Clothes Campaign"
    )
    st.markdown("---")
    st.caption("Built for MSc Applied Finance & Wealth Management  \nData Visualization & Analytics")

# =============================================================
# HEADLINE METRICS (top of page)
# =============================================================
latest = combined[combined['Year'] == combined['Year'].max()].iloc[0]
base = combined[combined['Year'] == 2000].iloc[0]

col1, col2, col3, col4 = st.columns(4)
col1.metric("US Apparel Prices",  f"+{latest['Apparel_Idx']-100:.0f}%", "since 2000")
col2.metric("US Overall CPI",     f"+{latest['Overall_Idx']-100:.0f}%", "since 2000")
col3.metric("BD Garment Wage",    "~$100/mo",  "≈ 20% of a living wage")
col4.metric("Worker's Share",     "1.2%",      "of a $10 t-shirt")

st.markdown("---")

# =============================================================
# TABS
# =============================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎯 The Divergence",
    "📉 Wage Freeze",
    "📊 Category Comparison",
    "🌍 Country Gap",
    "👕 T-Shirt Calculator",
    "🤖 ML Playground",
])

# -------- TAB 1: THE DIVERGENCE --------
with tab1:
    st.subheader("The Divergence")
    st.markdown(
        "US clothing prices barely moved since 2000 while overall CPI nearly doubled. "
        "Meanwhile, Bangladesh's real garment wage has crawled forward only through discrete political revisions."
    )

    df = combined[(combined['Year'] >= year_range[0]) & (combined['Year'] <= year_range[1])].copy()
    plot_df = df.melt(
        id_vars='Year',
        value_vars=['Apparel_Idx','Overall_Idx','BD_Real_Wage_Idx'],
        var_name='Series',
        value_name='Index',
    ).dropna()
    name_map = {
        'Apparel_Idx': 'US Apparel Prices',
        'Overall_Idx': 'US Overall CPI',
        'BD_Real_Wage_Idx': 'Bangladesh Real Wage',
    }
    plot_df['Series'] = plot_df['Series'].map(name_map)

    fig = px.line(
        plot_df, x='Year', y='Index', color='Series',
        color_discrete_map={
            'US Apparel Prices': '#c0392b',
            'US Overall CPI': '#2c3e50',
            'Bangladesh Real Wage': '#e67e22',
        },
        title="Cheap Clothes, Frozen Wages, Rising Everything Else (Indexed to 2000 = 100)",
    )
    fig.add_hline(y=100, line_dash="dot", line_color="gray", opacity=0.5)
    fig.update_layout(height=500, hovermode='x unified', legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("💡 What this chart shows"):
        st.markdown(
            "- **US Apparel Prices (red):** barely rose in 26 years — the only major category that didn't inflate.  \n"
            "- **US Overall CPI (navy):** nearly doubled — normal inflation for every other essential.  \n"
            "- **Bangladesh Real Wage (orange):** step changes on wage revisions, erosion between them.  \n"
            "  \n"
            "The story: someone absorbed the cost that consumers didn't pay. This chart shows who."
        )

# -------- TAB 2: WAGE FREEZE --------
with tab2:
    st.subheader("The Wage Freeze Timeline")
    st.markdown(
        "Bangladesh's RMG minimum wage looks like a series of victories in nominal taka. "
        "Deflate for inflation and each victory gets eaten before the next revision arrives."
    )

    bd_f = bd[(bd['Year'] >= year_range[0]) & (bd['Year'] <= year_range[1])]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=bd_f['Year'], y=bd_f['RMG_Wage_BDT'],
        name='Nominal Wage (BDT)',
        mode='lines', line=dict(color='#27ae60', width=3, shape='hv'),
        fill='tozeroy', fillcolor='rgba(39,174,96,0.15)',
    ))
    revisions = bd_f[bd_f['Is_Revision_Year'] == 1]
    fig.add_trace(go.Scatter(
        x=revisions['Year'], y=revisions['RMG_Wage_BDT'],
        mode='markers+text', name='Revision',
        marker=dict(size=14, color='#27ae60', line=dict(color='white', width=2)),
        text=[f"Tk {w:,.0f}" for w in revisions['RMG_Wage_BDT']],
        textposition="top center",
    ))
    fig.add_trace(go.Scatter(
        x=bd_f['Year'], y=bd_f['Real_Wage_2010BDT'],
        name='Real Wage (2010 BDT)',
        mode='lines', line=dict(color='#c0392b', width=2.5, dash='dash'),
        yaxis='y2',
    ))
    fig.update_layout(
        title="Every 'Raise' Gets Eaten by Inflation",
        xaxis_title='Year',
        yaxis=dict(title='Nominal Wage (BDT)', side='left', color='#27ae60'),
        yaxis2=dict(title='Real Wage (2010 BDT)', side='right', overlaying='y', color='#c0392b'),
        height=500, hovermode='x unified',
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    latest_bd = bd_f.dropna(subset=['Real_Wage_2010BDT']).iloc[-1]
    col1.metric("Current nominal wage", f"Tk {latest_bd['RMG_Wage_BDT']:,.0f}")
    col2.metric("Current real wage (2010 BDT)", f"Tk {latest_bd['Real_Wage_2010BDT']:,.0f}")
    col3.metric("USD equivalent", f"${latest_bd['Wage_USD_Market']:.0f}/mo")

# -------- TAB 3: CATEGORY COMPARISON --------
with tab3:
    st.subheader("Apparel vs Everything Else")
    st.markdown("Cumulative price change since 2000. Every essential category doubled. Only clothing stayed flat.")

    end_yr = year_range[1]
    df_cat = us[us['Year'].isin([2000, end_yr])].pivot(
        index='Category', columns='Year', values='Index_2000'
    ).reset_index()
    df_cat['Change_%'] = (df_cat[end_yr] / df_cat[2000] - 1) * 100
    df_cat = df_cat.sort_values('Change_%')

    fig = px.bar(
        df_cat, x='Category', y='Change_%',
        color='Change_%', color_continuous_scale=['#c0392b','#e67e22','#2c3e50'],
        text=df_cat['Change_%'].apply(lambda v: f"{v:+.1f}%"),
        title=f"Cumulative Price Change: 2000 → {end_yr}",
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(height=450, showlegend=False, yaxis_title="Change (%)", xaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    st.info("💭 **Ask yourself:** what economic force flattens one category while doubling all others?")

# -------- TAB 4: COUNTRY GAP --------
with tab4:
    st.subheader("The Living Wage Gap")
    st.markdown("Legal minimum vs estimated living wage in the top garment-exporting countries.")

    region_filter = st.multiselect(
        "Filter by region",
        options=sorted(countries['Region'].unique()),
        default=sorted(countries['Region'].unique()),
    )
    df_c = countries[countries['Region'].isin(region_filter)].sort_values('Coverage_%')

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df_c['Country'], x=df_c['Living_Wage_USD'],
        orientation='h', name='Living Wage', marker_color='#bdc3c7',
        text=[f"${v}" for v in df_c['Living_Wage_USD']], textposition='outside',
    ))
    fig.add_trace(go.Bar(
        y=df_c['Country'], x=df_c['Min_Wage_USD'],
        orientation='h', name='Legal Minimum', marker_color='#c0392b',
        text=[f"${v} ({c:.0f}% covered)" for v, c in zip(df_c['Min_Wage_USD'], df_c['Coverage_%'])],
        textposition='inside', textfont=dict(color='white'),
    ))
    fig.update_layout(
        title="Minimum Wage vs Living Wage (USD/month)",
        barmode='overlay', height=500,
        legend=dict(orientation="h", y=-0.1),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        df_c[['Country','Region','Regime','Min_Wage_USD','Living_Wage_USD','Coverage_%']],
        use_container_width=True, hide_index=True,
    )

# -------- TAB 5: T-SHIRT CALCULATOR --------
with tab5:
    st.subheader("The T-Shirt Cost Calculator")
    st.markdown("What if we paid garment workers more? Use the slider to see the impact on retail price.")

    col1, col2 = st.columns([1, 2])

    with col1:
        wage_multiplier = st.slider(
            "Multiply worker wage by:",
            min_value=1.0, max_value=10.0, value=1.0, step=0.5,
            help="A 2x multiplier doubles what the worker earns. What happens to the shirt price?",
        )
        pass_through = st.slider(
            "Pass-through to retail (%)",
            min_value=0, max_value=100, value=100, step=25,
            help="If brands absorb the increase, retail doesn't rise. If they pass it fully to consumers, retail rises 1:1.",
        ) / 100

        original_wage = tshirt.loc[tshirt['Component']=='Worker Wages','Cost_USD'].values[0]
        new_wage = original_wage * wage_multiplier
        wage_increase = new_wage - original_wage
        price_increase = wage_increase * pass_through
        new_price = 10 + price_increase

        st.markdown("### Impact")
        st.metric("Original t-shirt price", "$10.00")
        st.metric("New t-shirt price", f"${new_price:.2f}", f"+${price_increase:.2f}")
        st.metric("New worker wage", f"${new_wage:.2f}", f"+${wage_increase:.2f} ({(wage_multiplier-1)*100:.0f}%)")

    with col2:
        tshirt_new = tshirt.copy()
        tshirt_new.loc[tshirt_new['Component']=='Worker Wages','Cost_USD'] = new_wage
        # If pass-through < 100%, the remainder eats into brand margin
        absorbed = wage_increase * (1 - pass_through)
        tshirt_new.loc[tshirt_new['Component']=='Brand Margin','Cost_USD'] -= absorbed

        fig = go.Figure()
        colors = ['#c0392b','#e67e22','#f39c12','#95a5a6','#3498db','#2c3e50']
        cum = 0
        for i, row in tshirt_new.iterrows():
            fig.add_trace(go.Bar(
                x=[row['Component']], y=[row['Cost_USD']],
                base=[cum], marker_color=colors[i],
                text=[f"${row['Cost_USD']:.2f}"], textposition='inside',
                textfont=dict(color='white'), showlegend=False,
                name=row['Component'],
            ))
            cum += row['Cost_USD']
        fig.update_layout(
            title=f"T-Shirt Breakdown (Total: ${cum:.2f})",
            yaxis_title='Cost (USD)', height=450, barmode='stack',
        )
        st.plotly_chart(fig, use_container_width=True)

    st.success(
        f"💡 **Key insight:** paying garment workers **{wage_multiplier:.1f}× more** raises the shirt price by "
        f"only **${price_increase:.2f}** — a **{(price_increase/10)*100:.1f}%** change in retail. "
        f"Ethical pricing is arithmetically trivial."
    )

# -------- TAB 6: ML PLAYGROUND --------
with tab6:
    st.subheader("K-Means Clustering Playground")
    st.markdown("Choose the number of clusters and see how garment-exporting countries group by wage regime.")

    col1, col2 = st.columns([1, 3])

    with col1:
        k = st.slider("Number of clusters (k)", 2, 6, 3)
        random_state = st.number_input("Random seed", value=42)
        st.markdown("---")
        st.markdown("**Features used:**")
        st.caption("• Min Wage (USD)  \n• Living Wage (USD)  \n• Coverage %")

    with col2:
        X = countries[['Min_Wage_USD','Living_Wage_USD','Coverage_%']].values
        X_scaled = StandardScaler().fit_transform(X)
        km = KMeans(n_clusters=k, random_state=int(random_state), n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)

        countries_view = countries.copy()
        countries_view['Cluster'] = [f"Cluster {i}" for i in labels]

        fig = px.scatter(
            countries_view, x='Min_Wage_USD', y='Living_Wage_USD',
            color='Cluster', text='Country', size='Gap_USD',
            title=f"K-Means Clustering (k={k}, Silhouette Score = {sil:.3f})",
            hover_data=['Coverage_%','Regime'],
        )
        max_v = max(countries['Min_Wage_USD'].max(), countries['Living_Wage_USD'].max())
        fig.add_trace(go.Scatter(
            x=[0, max_v], y=[0, max_v],
            mode='lines', line=dict(dash='dash', color='gray'),
            name='100% coverage', showlegend=True,
        ))
        fig.update_traces(textposition='top center', selector=dict(mode='markers+text'))
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Silhouette Score Comparison")
    ks = list(range(2, 7))
    scores = []
    for kk in ks:
        km2 = KMeans(n_clusters=kk, random_state=42, n_init=10).fit(X_scaled)
        scores.append(silhouette_score(X_scaled, km2.labels_))
    fig2 = px.bar(x=ks, y=scores, labels={'x':'k','y':'Silhouette Score'},
                  title="Best k by Silhouette (higher = better-separated clusters)",
                  color=scores, color_continuous_scale='RdYlGn')
    fig2.update_layout(height=300, showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

    best_k = ks[np.argmax(scores)]
    st.info(
        f"💡 **Grid search finding:** k = {best_k} gives the best-separated clusters "
        f"(silhouette = {max(scores):.3f}). This suggests the world of garment production "
        f"splits naturally into **{best_k} groups**, not more."
    )

# =============================================================
# FOOTER
# =============================================================
st.markdown("---")
st.caption(
    "**The Hidden Ledger** — a fast fashion analytics project | "
    "MSc Applied Finance & Wealth Management | "
    "Data: FRED, World Bank, Asia Floor Wage Alliance | "
    "Built with Streamlit + Plotly + scikit-learn"
)
