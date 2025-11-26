import streamlit as st
import time
import json
import re
import os
import pandas as pd
import altair as alt
from datetime import datetime
from openai import OpenAI

# ==========================================
# 0. СЕКРЕТНЫЕ НАСТРОЙКИ
# ==========================================
import os

# --- ВСТАВЬТЕ ЭТОТ БЛОК ---
try:
    API_KEY = st.secrets["API_KEY"]
except:
    API_KEY = None
# ---------------------------

DB_FILE = "session_database.json"
# ==========================================
# 1. КОНФИГУРАЦИЯ И ДИЗАЙН
# ==========================================
st.set_page_config(layout="wide", page_title="PsyCounAssist: Pro")

st.markdown("""
<style>
    .stApp { background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%); color: #e2e8f0; font-family: 'Inter', sans-serif; }
    .block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; }
    header, footer { visibility: hidden; }
    [data-testid="stSidebar"] { background-color: #0b1120; border-right: 1px solid #1e293b; }

    .stChatMessage { background-color: rgba(30, 41, 59, 0.7); border: 1px solid #334155; border-radius: 12px; padding: 15px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
    .stChatMessage:hover { border-color: #64748b; background-color: rgba(30, 41, 59, 1); }
    .stChatMessage .stAvatar { background-color: #3b82f6; }

    .monitor-container { background: rgba(15, 23, 42, 0.6); backdrop-filter: blur(10px); border-radius: 16px; border: 1px solid #334155; padding: 20px; }
    .metric-card { background: #1e293b; border-left: 4px solid #3b82f6; border-radius: 8px; padding: 12px; margin-bottom: 12px; }
    .metric-title { color: #94a3b8; font-size: 10px; text-transform: uppercase; letter-spacing: 1px; font-weight: 700; margin-bottom: 4px; }
    .metric-value { color: #f1f5f9; font-size: 14px; font-weight: 500; }

    .custom-tag { display: inline-block; padding: 4px 10px; margin: 3px; border-radius: 20px; font-size: 11px; font-weight: 600; background: #334155; color: #e2e8f0; border: 1px solid #475569; }
    .tag-highlight { background: #1e1b4b; color: #a5b4fc; border-color: #4338ca; }

    div.stButton > button, div.stDownloadButton > button { background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%); color: white; border: none; border-radius: 8px; padding: 12px 20px; font-weight: 600; width: 100%; transition: all 0.2s ease; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2); }
    div.stButton > button:hover, div.stDownloadButton > button:hover { transform: scale(1.02); box-shadow: 0 0 20px rgba(59, 130, 246, 0.6); color: white; border: none; }

    .json-box { font-family: 'JetBrains Mono', monospace; font-size: 10px; line-height: 1.4; }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 2. PERSISTENCE (СОХРАНЕНИЕ)
# ==========================================
def load_session():
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return None
    return None


def save_session():
    data = {
        "history": st.session_state.history,
        "chart_data": st.session_state.chart_data,
        "last_analysis": st.session_state.last_analysis,
        "msg_count": st.session_state.msg_count
    }
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def reset_session():
    if os.path.exists(DB_FILE): os.remove(DB_FILE)
    st.session_state.history = []
    st.session_state.chart_data = []
    st.session_state.msg_count = 0
    st.session_state.last_analysis = {
        "hypothesis": "Сбор данных...", "triggers": [],
        "recommendations": [], "sentiment": 0, "status": "N/A"
    }
    st.rerun()


if "history" not in st.session_state:
    saved = load_session()
    if saved:
        st.session_state.history = saved["history"]
        st.session_state.chart_data = saved["chart_data"]
        st.session_state.last_analysis = saved["last_analysis"]
        st.session_state.msg_count = saved.get("msg_count", 0)
    else:
        st.session_state.history = []
        st.session_state.chart_data = []
        st.session_state.msg_count = 0
        st.session_state.last_analysis = {
            "hypothesis": "Сбор данных...", "triggers": [],
            "recommendations": [], "sentiment": 0, "status": "N/A"
        }


# ==========================================
# 3. МАТЕМАТИЧЕСКОЕ ЯДРО (СГЛАЖИВАНИЕ)
# ==========================================
class MathEngine:
    @staticmethod
    def calculate_smooth_sentiment(current_val, target_val):
        """
        Алгоритм инерции. Не дает графику прыгать.
        Максимальный сдвиг за шаг ~0.1 (или 0.2 если сильный позитив).
        """
        delta = target_val - current_val

        # Определяем лимит изменения (Clamp)
        # Если эмоция резко положительная (delta > 0), разрешаем чуть больший скачок
        if delta > 0:
            max_step = 0.25  # Позволяем рост до +0.25
        else:
            max_step = 0.15  # Падение ограничиваем жестче (-0.15)

        # Ограничиваем изменение
        if delta > max_step: delta = max_step
        if delta < -max_step: delta = -max_step

        # Новое значение
        new_val = current_val + delta

        # Округляем и держим в границах -1..1
        return round(max(-1.0, min(1.0, new_val)), 2)


# ==========================================
# 4. АРХИТЕКТУРНОЕ ЯДРО (DEEPSEEK)
# ==========================================
class NeuralCore:
    def __init__(self):
        self.base_url = "https://api.vsegpt.ru/v1"
        self.client = OpenAI(api_key=API_KEY, base_url=self.base_url) if API_KEY else None

        self.system_prompt = """
        Ты — PsyCounAssist. Веди теплый диалог (КПТ).

        ОБЯЗАТЕЛЬНО В КАЖДОМ ОТВЕТЕ ПИШИ JSON:
        <analysis>
        {
            "sentiment": 0.0, (Оцени от -1.0 до 1.0)
            "hypothesis": "Гипотеза",
            "triggers": ["Т1", "Т2"],
            "recommendations": ["Р1", "Р2"],
            "status": "Статус"
        }
        </analysis>
        """

    def process(self, text, history):
        if not self.client: return "NO_KEY", "⚠️ API Key не найден."

        messages = [{"role": "system", "content": self.system_prompt}]
        for msg in history:
            if msg["role"] != "system": messages.append(msg)
        messages.append({"role": "user", "content": text})

        try:
            resp = self.client.chat.completions.create(
                model="google/gemini-3-pro-preview-high",
                messages=messages, temperature=0.7
            )
            return "OK", resp.choices[0].message.content
        except Exception as e:
            return "ERROR", str(e)

    @staticmethod
    def parse_and_update(raw):
        match = re.search(r'<analysis>(.*?)</analysis>', raw, re.DOTALL)
        clean = re.sub(r'<analysis>.*?</analysis>', '', raw, flags=re.DOTALL).strip()

        new_data = None
        if match:
            try:
                raw_json = json.loads(match.group(1))

                # --- ЛОГИКА ОБНОВЛЕНИЯ (GATING) ---
                # 1. Берем последнее известное настроение (или 0)
                last_sentiment = 0
                if len(st.session_state.chart_data) > 0:
                    last_sentiment = st.session_state.chart_data[-1]['sentiment']

                # 2. Считаем новое плавное настроение
                target_sentiment = raw_json.get('sentiment', 0)
                smooth_sentiment = MathEngine.calculate_smooth_sentiment(last_sentiment, target_sentiment)

                # 3. Формируем объект для обновления
                new_data = {
                    "sentiment": smooth_sentiment,  # Всегда обновляем график
                    "status": raw_json.get("status", "N/A")
                }

                # 4. Проверяем счетчик: обновлять ли диагноз?
                # Обновляем гипотезу только на 1-м сообщении и каждом 5-м (1, 5, 10...)
                is_full_update = (st.session_state.msg_count == 1) or (st.session_state.msg_count % 5 == 0)

                if is_full_update:
                    # Полное обновление
                    new_data["hypothesis"] = raw_json.get("hypothesis", "...")
                    new_data["triggers"] = raw_json.get("triggers", [])
                    new_data["recommendations"] = raw_json.get("recommendations", [])
                    update_type = "FULL"
                else:
                    # Частичное (берем старые значения)
                    old = st.session_state.last_analysis
                    new_data["hypothesis"] = old.get("hypothesis", "...")
                    new_data["triggers"] = old.get("triggers", [])
                    new_data["recommendations"] = old.get("recommendations", [])
                    update_type = "SENTIMENT_ONLY"

                return clean, new_data, update_type

            except:
                pass
        return clean, None, "ERROR"


# ==========================================
# 5. ИНТЕРФЕЙС
# ==========================================
engine = NeuralCore()

# --- САЙДБАР ---
with st.sidebar:
    st.markdown("### 🛠️ ПАНЕЛЬ РАЗРАБОТЧИКА")
    if st.button("HARD RESET", use_container_width=True): reset_session()
    st.divider()

    # Индикатор цикла
    cycle = st.session_state.msg_count % 5
    if cycle == 0 and st.session_state.msg_count > 0: cycle = 5

    st.markdown(f"**АНАЛИЗ ЦИКЛА:** {cycle}/5")
    st.progress(cycle / 5)

    if cycle == 5 or st.session_state.msg_count == 1:
        st.success("⚡ ВЫВОД")
    else:
        st.info("⚡ ВЫВОД")

    st.divider()
    st.caption(f"System Status: {'ONLINE' if API_KEY else 'OFFLINE'}")

# --- MAIN ---
col_chat, col_dash = st.columns([0.65, 0.35], gap="large")

with col_chat:
    st.markdown("### ☁️ Сессия")
    chat_container = st.container(height=650)
    with chat_container:
        if not st.session_state.history:
            st.markdown("""
            <div style='text-align: center; color: #64748b; margin-top: 50px;'>
                <h3>Привет! Я рядом.</h3>
                <p>Я здесь, чтобы выслушать и помочь.</p>
            </div>
            """, unsafe_allow_html=True)
        for msg in st.session_state.history:
            avatar = "👤" if msg['role'] == 'user' else "🤖"
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])

    if prompt := st.chat_input("Напишите сообщение..."):
        st.session_state.history.append({"role": "user", "content": prompt})
        st.session_state.msg_count += 1  # +1 сообщение

        with chat_container:
            with st.chat_message("user", avatar="👤"):
                st.markdown(prompt)
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Анализ..."):
                    status, raw = engine.process(prompt, st.session_state.history)
                    if status == "OK":
                        text, data, update_type = engine.parse_and_update(raw)

                        if data:
                            st.session_state.last_analysis = data
                            st.session_state.chart_data.append({
                                "step": len(st.session_state.chart_data),
                                "sentiment": data['sentiment'],
                                "status": data['status']
                            })

                        st.markdown(text)
                        st.session_state.history.append({"role": "assistant", "content": text})
                        save_session()
                    else:
                        st.error(raw)
        st.rerun()

with col_dash:
    st.markdown("### 🩺 Клинический Монитор")
    data = st.session_state.last_analysis

    # Метрики
    st.markdown(f"""
    <div class="monitor-container">
        <div class="metric-card">
            <div class="metric-title">Текущая Гипотеза</div>
            <div class="metric-value">{data.get('hypothesis', '...')}</div>
        </div>
        <div style="display: flex; gap: 10px;">
             <div class="metric-card" style="flex: 1; border-color: #a855f7;">
                <div class="metric-title">Статус</div>
                <div class="metric-value">{data.get('status', 'N/A')}</div>
            </div>
            <div class="metric-card" style="flex: 1; border-color: #10b981;">
                <div class="metric-title">Эмоц.сост.</div>
                <div class="metric-value">{data.get('sentiment', 0)}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ГРАФИК
    st.markdown("#### 📈 Динамика")
    if len(st.session_state.chart_data) > 0:
        df = pd.DataFrame(st.session_state.chart_data)

        chart = alt.Chart(df).mark_area(
            line={'color': '#3b82f6'},
            color=alt.Gradient(
                gradient='linear',
                stops=[alt.GradientStop(color='#3b82f6', offset=0),
                       alt.GradientStop(color='rgba(59, 130, 246, 0)', offset=1)],
                x1=1, x2=1, y1=1, y2=0
            )
        ).encode(
            x=alt.X('step', axis=None),
            y=alt.Y('sentiment',
                    scale=alt.Scale(domain=[-1, 1]),
                    title='Эмоц. сост.',
                    axis=alt.Axis(titleColor='#94a3b8', labelColor='#94a3b8', gridColor='#334155')
                    ),
            tooltip=['status', 'sentiment']
        ).properties(height=200)

        rule = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color='#64748b', strokeDash=[2, 2]).encode(y='y')

        final_chart = (chart + rule).configure_view(strokeWidth=0).configure(background='transparent')
        st.altair_chart(final_chart, use_container_width=True)
    else:
        st.info("График строится в реальном времени...")

    # ПАТТЕРНЫ
    st.markdown("#### 🧩 Паттерны")
    if data.get('triggers'):
        st.markdown("**Триггеры:**")
        html_trig = "".join([f"<span class='custom-tag tag-highlight'>{t}</span>" for t in data['triggers']])
        st.markdown(html_trig, unsafe_allow_html=True)

    if data.get('recommendations'):
        st.markdown("**Протоколы:**", unsafe_allow_html=True)
        html_rec = "".join([f"<span class='custom-tag'>{r}</span>" for r in data['recommendations']])
        st.markdown(html_rec, unsafe_allow_html=True)

    st.divider()

    report_text = f"""
    CLINICAL REPORT
    Date: {datetime.now().strftime("%d.%m.%Y")}
    Hypothesis: {data.get('hypothesis')}
    Sentiment: {data.get('sentiment')}
    Triggers: {', '.join(data.get('triggers', []))}
    Recommendations: {', '.join(data.get('recommendations', []))}
    """

    st.download_button(
        label="📄 Сформировать PDF-отчет",
        data=report_text,
        file_name=f"report.txt",
        mime="text/plain",
        use_container_width=True

    )



