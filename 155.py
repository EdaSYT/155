import streamlit as st
import pandas as pd
from gurobipy import Model, GRB, quicksum

# Streamlit Sayfa Ayarları
st.set_page_config(layout="wide", page_title="Hattı Dengeleme Optimizasyonu")
st.title("🏭 Montaj Hattı Dengeleme & Operatör Atama Sistemi (Gurobi Engine)")

# =========================================================
# SABİT VERİLER
# =========================================================
I = range(1, 64)
J = range(1, 37)
W = range(1, 37)

t = {
    1: 2.43,  2: 9.79,  3: 2.12,  4: 9.92,  5: 4.66,  6: 11.58, 7: 1.01,  8: 1.44,  9: 9.66, 10: 10.30, 
    11: 0.49, 12: 7.13, 13: 7.18, 14: 2.44, 15: 3.58, 16: 4.90, 17: 3.21, 18: 7.78, 19: 11.27, 20: 11.35, 
    21: 0.80, 22: 3.31, 23: 9.83, 24: 0.80, 25: 4.61, 26: 5.20, 27: 11.89, 28: 6.30, 29: 13.32, 30: 0.98,
    31: 14.20, 32: 6.13, 33: 0.98, 34: 14.49, 35: 3.14, 36: 12.12, 37: 1.07, 38: 5.14, 39: 5.63, 40: 0.57, 
    41: 10.13, 42: 0.90, 43: 1.39, 44: 1.43, 45: 0.51, 46: 10.74, 47: 5.65, 48: 7.38, 49: 1.71, 50: 15.09, 
    51: 7.31, 52: 6.93, 53: 10.72, 54: 1.31, 55: 6.45, 56: 2.39, 57: 0.89, 58: 11.06, 59: 8.02, 60: 6.48,
    61: 3.13, 62: 0.53, 63: 7.74
}

P = [(i, i + 1) for i in range(1, 63)]
d_dist = {j: {k: 2 * abs(j - k) for k in range(1, 38)} for j in range(1, 38)}
BIG_M = sum(t.values())

# =========================================================
# YAN MENÜ (SIDEBAR)
# =========================================================
with st.sidebar:
    st.header("⚙️ Ayarlar")
    with st.expander("🏗️ Hat Parametreleri", expanded=True):
        L = st.number_input("Maksimum Yürüme Mesafesi (L)", value=4)
        D_target = st.number_input("Hedef Üretim Miktarı (D)", value=32)
        T_shift = st.number_input("Vardiya Süresi (T - dk)", value=510)
    
    with st.expander("⚖️ Optimizasyon Kısıtları", expanded=True):
        U_MAX = st.slider("Maks. Operatör Doluluğu (U_MAX)", 0.1, 1.0, 1.0, step=0.05)
    
    st.markdown("---")
    target_workers = st.slider("Detaylı Rapor İçin Operatör Seç", 1, 36, 29)

# =========================================================
# GUROBI MODEL ÇÖZÜCÜ
# =========================================================
@st.cache_data(show_spinner=False)
def solve_gurobi(exact_workers, L_val, D_val, T_val, U_limit):
    m = Model("st_balancing")
    m.setParam("OutputFlag", 0)
    m.setParam("TimeLimit", 10) # Her senaryo için max 10 saniye

    # Karar Değişkenleri
    x = m.addVars(I, J, vtype=GRB.BINARY, name="x")
    y = m.addVars(W, J, vtype=GRB.BINARY, name="y")
    z = m.addVars(W, vtype=GRB.BINARY, name="z")
    l = m.addVars(J, lb=0.0, vtype=GRB.CONTINUOUS, name="l")
    q = m.addVars(W, J, lb=0.0, vtype=GRB.CONTINUOUS, name="q")
    C = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="C")

    # Kısıtlar
    for i in I: m.addConstr(quicksum(x[i, j] for j in J) == 1)
    for i, h in P: m.addConstr(quicksum(j * x[i, j] for j in J) <= quicksum(j * x[h, j] for j in J))
    for j in J: m.addConstr(l[j] == quicksum(t[i] * x[i, j] for i in I))
    for j in J: m.addConstr(quicksum(y[w, j] for w in W) == 1)
    for w in W:
        for j in J: m.addConstr(y[w, j] <= z[w])
    
    # Linearize q[w,j]
    for w in W:
        for j in J:
            m.addConstr(q[w, j] <= l[j])
            m.addConstr(q[w, j] <= BIG_M * y[w, j])
            m.addConstr(q[w, j] >= l[j] - BIG_M * (1 - y[w, j]))
            
    for w in W: m.addConstr(quicksum(q[w, j] for j in J) <= C)
    for j in J: m.addConstr(l[j] <= C)
    
    # Doluluk Kısıtı (U_MAX)
    for w in W:
        m.addConstr((D_val / T_val) * quicksum(q[w, j] for j in J) <= U_limit)

    # Mesafe Kısıtı
    for w in W:
        for j in J:
            for k in J:
                if j < k and d_dist[j][k] > L_val:
                    m.addConstr(y[w, j] + y[w, k] <= 1)

    m.addConstr(quicksum(z[w] for w in W) == exact_workers)
    
    m.setObjective(C, GRB.MINIMIZE)
    m.optimize()

    if m.status == GRB.OPTIMAL or m.status == GRB.FEASIBLE:
        c_val = C.X
        return {
            "C": c_val,
            "ops_of_station": {j: [i for i in I if x[i, j].X > 0.5] for j in J},
            "stations_of_worker": {w: [j for j in J if y[w, j].X > 0.5] for w in W},
            "station_loads": {j: l[j].X for j in J},
            "worker_U": {w: 100 * (D_val / T_val) * sum(q[w, j].X for j in J) for w in W},
            "reachable_output": T_val / c_val if c_val > 1e-6 else 0,
            "meets_target": (T_val / c_val >= D_val - 1e-6)
        }
    return None

# =========================================================
# ANA PANEL
# =========================================================
st.subheader(f"📊 Analiz Sonuçları (U_MAX: %{U_MAX*100:.0f})")

if st.button("🚀 Senaryoları Gurobi ile Hesapla"):
    with st.spinner("Gurobi motoru çalışıyor..."):
        summary_data = []
        detailed_results = {}
        
        for n in range(1, 37):
            res = solve_gurobi(n, L, D_target, T_shift, U_MAX)
            if res:
                summary_data.append([
                    n, f"{res['C']:.2f} dk", f"{len([j for j in J if res['station_loads'][j]>0])} İst.",
                    f"{res['reachable_output']:.2f} Adet", "✅" if res['meets_target'] else "⚠️ Düşük"
                ])
                detailed_results[n] = res
            else:
                summary_data.append([n, "Çözüm Yok", "-", "-", "❌"])
                detailed_results[n] = None

        df_summary = pd.DataFrame(summary_data, columns=["İşçi Sayısı", "Çevrim Süresi", "İstasyon", "Kapasite", "Durum"])
        st.dataframe(df_summary, use_container_width=True, hide_index=True)

        st.divider()
        res_target = detailed_results.get(target_workers)
        if res_target:
            st.success(f"🎯 {target_workers} Operatör İçin Detaylı Dağılım")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**İstasyon Yükleri**")
                s_list = [[j, ", ".join(map(str, res_target['ops_of_station'][j])), f"{res_target['station_loads'][j]:.2f} dk"] 
                          for j in J if res_target['station_loads'][j] > 0]
                st.table(pd.DataFrame(s_list, columns=["No", "Operasyonlar", "Süre"]))
            with col2:
                st.markdown("**Operatör Dolulukları**")
                w_list = [[w, ", ".join(map(str, res_target['stations_of_worker'][w])), f"%{res_target['worker_U'][w]:.2f}"] 
                          for w in W if res_target['stations_of_worker'][w]]
                st.table(pd.DataFrame(w_list, columns=["Op", "Sorumlu İstasyonlar", "Verimlilik"]))
        else:
            st.error(f"❌ {target_workers} İşçi için uygun çözüm bulunamadı.")
else:
    st.info("Hesaplamayı başlatmak için 'Senaryoları Gurobi ile Hesapla' butonuna basın.")
