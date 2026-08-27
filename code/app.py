# -*- coding: utf-8 -*-
"""
DIPAS - Diseño Inverso para Perfiles con Autoencoder y Simulación
Interfaz Gráfica Interactiva con Streamlit y Plotly
Paleta: Dark Amethyst (#0D0630), Deep Space Blue (#18314F), Dusk Blue (#384E77), Tropical Teal (#00AFB5), White (#FFFFFF)
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import importlib
import dipas_engine
importlib.reload(dipas_engine)
from dipas_engine import DIPASEngine

# ==============================================================================
# CONFIGURACIÓN DE PÁGINA Y ESTILOS CSS
# ==============================================================================
# ==============================================================================
# CONFIGURACIÓN DE PÁGINA Y ESTILOS CSS (ACADEMIC SCIENTIFIC LIGHT THEME)
# ==============================================================================
st.set_page_config(
    page_title="DIPAS — Inverse Airfoil Design",
    page_icon="✈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# CONFIGURACIÓN DINÁMICA DE ESTILOS CSS (LIGHT / DARK PALETTE)
# ==============================================================================
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

is_dark = st.session_state.dark_mode

if is_dark:
    # PALETA OSCURA DIPAS: Dark Amethyst (#0D0630), Deep Space Blue (#18314F), Dusk Blue (#384E77), Tropical Teal (#00AFB5)
    theme_vars = """
        --academic-navy: #00AFB5 !important;
        --academic-blue: #384E77 !important;
        --tropical-teal: #00AFB5 !important;
        --teal-dark: #008B90 !important;
        --paper-bg: #0D0630 !important;
        --card-bg: #18314F !important;
        --sidebar-bg: #11093B !important;
        --input-bg: #18314F !important;
        --tab-list-bg: #18314F !important;
        --tab-active-bg: #0D0630 !important;
        --tab-text: #CBD5E1 !important;
        --border-color: #384E77 !important;
        --border-dark: #384E77 !important;
        --text-primary: #FFFFFF !important;
        --text-secondary: #CBD5E1 !important;
        --text-muted: #94A3B8 !important;
        --badge-bg: #18314F !important;
        --badge-color: #00AFB5 !important;
        --header-border: #00AFB5 !important;
        --background-color: #0D0630 !important;
        --secondary-background-color: #18314F !important;
        --text-color: #FFFFFF !important;
        --primary-color: #00AFB5 !important;
    """
else:
    # PALETA CLARA CIENTÍFICA: Paper White (#F8FAFC), Card White (#FFFFFF), Navy (#0F2C59), Teal (#00AFB5)
    theme_vars = """
        --academic-navy: #0F2C59 !important;
        --academic-blue: #1E3A8A !important;
        --tropical-teal: #00AFB5 !important;
        --teal-dark: #008B90 !important;
        --paper-bg: #F8FAFC !important;
        --card-bg: #FFFFFF !important;
        --sidebar-bg: #FFFFFF !important;
        --input-bg: #FFFFFF !important;
        --tab-list-bg: #EEF2F6 !important;
        --tab-active-bg: #FFFFFF !important;
        --tab-text: #1E293B !important;
        --border-color: #E2E8F0 !important;
        --border-dark: #CBD5E1 !important;
        --text-primary: #0F172A !important;
        --text-secondary: #334155 !important;
        --text-muted: #64748B !important;
        --badge-bg: #F1F5F9 !important;
        --badge-color: #0F2C59 !important;
        --header-border: #0F2C59 !important;
        --background-color: #F8FAFC !important;
        --secondary-background-color: #FFFFFF !important;
        --text-color: #0F172A !important;
        --primary-color: #00AFB5 !important;
    """

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Lora:ital,wght@0,400;0,500;0,600;0,700;1,400&family=STIX+Two+Text:ital,wght@0,400;0,600;0,700;1,400;1,600&display=swap');

    :root,
    .stApp,
    [data-testid="stAppViewContainer"],
    section[data-testid="stSidebar"] {
        __THEME_VARS__
        --accent-amber: #D97706;
        --accent-emerald: #059669;
        --accent-crimson: #DC2626;
    }

    /* Header superior transparente permitiendo el botón de abrir/cerrar sidebar */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
        z-index: 99 !important;
    }

    /* Botón flotante para reabrir la barra lateral cuando está contraída */
    button[data-testid="stSidebarCollapsedControl"],
    div[data-testid="stSidebarCollapsedControl"] button,
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapseButton"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        background-color: var(--card-bg) !important;
        border: 1.5px solid var(--tropical-teal) !important;
        border-radius: 8px !important;
        color: var(--academic-navy) !important;
        box-shadow: 0 2px 8px rgba(0, 175, 181, 0.25) !important;
        transition: all 0.2s ease !important;
        z-index: 999999 !important;
    }
    button[data-testid="stSidebarCollapsedControl"]:hover,
    div[data-testid="stSidebarCollapsedControl"] button:hover,
    [data-testid="collapsedControl"]:hover,
    [data-testid="stSidebarCollapseButton"]:hover {
        background-color: var(--paper-bg) !important;
        border-color: var(--teal-dark) !important;
        box-shadow: 0 4px 12px rgba(0, 175, 181, 0.40) !important;
        transform: scale(1.05);
    }
    button[data-testid="stSidebarCollapsedControl"] svg,
    div[data-testid="stSidebarCollapsedControl"] svg,
    [data-testid="collapsedControl"] svg,
    [data-testid="stSidebarCollapseButton"] svg {
        fill: var(--tropical-teal) !important;
        stroke: var(--tropical-teal) !important;
    }

    /* Margen superior limpio para que nada se corte */
    .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 98% !important;
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 1.5rem !important;
        padding-left: 1.0rem !important;
        padding-right: 1.0rem !important;
    }

    /* Fondo general */
    .stApp {
        background-color: var(--paper-bg) !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Barra lateral */
    section[data-testid="stSidebar"] {
        background-color: var(--sidebar-bg) !important;
        border-right: 1px solid var(--border-color) !important;
    }
    
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span {
        color: var(--text-secondary);
    }
    
    section[data-testid="stSidebar"] .stMarkdown h1, 
    section[data-testid="stSidebar"] .stMarkdown h2, 
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: var(--academic-navy) !important;
        font-family: 'STIX Two Text', 'Lora', Georgia, serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.2px !important;
        margin-top: 0 !important;
    }

    /* Tarjetas y Contenedores Científicos */
    .dipas-card {
        background-color: var(--card-bg) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        color: var(--text-primary) !important;
    }
    .dipas-card h4, .dipas-card h5 {
        color: var(--academic-navy) !important;
    }
    .dipas-card p {
        color: var(--text-secondary) !important;
    }
    
    .dipas-card-accent {
        background-color: var(--card-bg) !important;
        border: 1px solid var(--border-color) !important;
        border-left: 4px solid var(--tropical-teal) !important;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        color: var(--text-primary) !important;
    }
    .dipas-card-accent h4, .dipas-card-accent h5 {
        color: var(--academic-navy) !important;
    }
    .dipas-card-accent p {
        color: var(--text-secondary) !important;
    }

    /* Métricas KPI */
    .kpi-container {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        background-color: var(--card-bg) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px;
        padding: 10px 8px;
        text-align: center;
        min-height: 100px;
        height: 100%;
        box-sizing: border-box;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
        transition: all 0.2s ease;
    }
    .kpi-container:hover {
        border-color: var(--tropical-teal) !important;
        box-shadow: 0 2px 6px rgba(0, 175, 181, 0.20);
    }
    .kpi-title {
        font-family: 'Inter', sans-serif;
        font-size: 0.70rem;
        font-weight: 600;
        color: var(--text-muted) !important;
        text-transform: uppercase;
        letter-spacing: 0.3px;
        white-space: normal;
        margin-bottom: 2px;
        line-height: 1.2;
    }
    .kpi-value {
        font-family: 'STIX Two Text', 'Lora', Georgia, serif;
        font-size: 1.40rem;
        font-weight: 700;
        color: var(--academic-navy) !important;
        margin: 2px 0;
        white-space: nowrap;
    }
    .kpi-sub {
        font-family: 'Inter', sans-serif;
        font-size: 0.72rem;
        color: var(--text-secondary) !important;
        white-space: normal;
        line-height: 1.25;
        word-break: break-word;
    }

    /* Pestañas (Tabs) */
    div[data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 8px !important;
        background-color: var(--tab-list-bg) !important;
        padding: 6px !important;
        border-radius: 8px !important;
        border: 1px solid var(--border-dark) !important;
        display: flex !important;
        flex-wrap: wrap !important;
    }

    div[data-testid="stTabs"] button[data-baseweb="tab"],
    div[data-testid="stTabs"] button[role="tab"] {
        background-color: transparent !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 8px 16px !important;
    }

    div[data-testid="stTabs"] button[data-baseweb="tab"] p,
    div[data-testid="stTabs"] button[data-baseweb="tab"] span,
    div[data-testid="stTabs"] button[data-baseweb="tab"] div,
    div[data-testid="stTabs"] button[role="tab"] p,
    div[data-testid="stTabs"] button[role="tab"] span,
    div[data-testid="stTabs"] button[role="tab"] div {
        color: var(--tab-text) !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.90rem !important;
        opacity: 1 !important;
    }

    div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"],
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        background-color: var(--tab-active-bg) !important;
        border: 1px solid var(--tropical-teal) !important;
        border-top: 3px solid var(--tropical-teal) !important;
        box-shadow: 0 1px 4px rgba(0, 175, 181, 0.15) !important;
    }

    div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] p,
    div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] span,
    div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] div,
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] p,
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] span,
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] div {
        color: var(--tropical-teal) !important;
        font-weight: 700 !important;
    }

    div[data-testid="stTabs"] div[data-baseweb="tab-highlight"] {
        background-color: var(--tropical-teal) !important;
    }
    div[data-testid="stTabs"] div[data-baseweb="tab-border"] {
        display: none !important;
    }

    /* Botones primarios y de descarga */
    .stButton > button,
    .stDownloadButton > button,
    div[data-testid="stDownloadButton"] > button,
    div[data-testid="stDownloadButton"] button,
    section[data-testid="stSidebar"] .stButton > button,
    button[data-testid="baseButton-secondary"] {
        background: linear-gradient(135deg, #00AFB5 0%, #008B90 100%) !important;
        background-color: #00AFB5 !important;
        color: #FFFFFF !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.90rem !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        letter-spacing: 0.3px !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 8px rgba(0, 175, 181, 0.30) !important;
    }
    
    .stButton > button:hover,
    .stDownloadButton > button:hover,
    div[data-testid="stDownloadButton"] > button:hover,
    section[data-testid="stSidebar"] .stButton > button:hover,
    button[data-testid="baseButton-secondary"]:hover {
        background: linear-gradient(135deg, #009DA3 0%, #007A7E 100%) !important;
        box-shadow: 0 4px 14px rgba(0, 175, 181, 0.50) !important;
        transform: translateY(-1px);
    }

    .stButton > button *,
    .stButton > button p,
    .stButton > button span,
    .stButton > button div,
    .stDownloadButton > button *,
    .stDownloadButton > button p,
    .stDownloadButton > button span,
    .stDownloadButton > button div,
    div[data-testid="stDownloadButton"] > button *,
    div[data-testid="stDownloadButton"] > button p,
    div[data-testid="stDownloadButton"] > button span,
    div[data-testid="stDownloadButton"] > button div,
    section[data-testid="stSidebar"] .stButton > button *,
    section[data-testid="stSidebar"] .stButton > button p,
    section[data-testid="stSidebar"] .stButton > button span,
    section[data-testid="stSidebar"] .stButton > button div,
    button[data-testid="baseButton-secondary"] *,
    button[data-testid="baseButton-secondary"] p,
    button[data-testid="baseButton-secondary"] span {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    /* Enlace terciario */
    div[class*="st-key-back_step1"] button,
    div[data-testid="stElementContainer"]:has([class*="st-key-back_step1"]) button,
    section[data-testid="stSidebar"] div[class*="st-key-back_step1"] button,
    section[data-testid="stSidebar"] button[data-testid="baseButton-tertiary"],
    button[data-testid="baseButton-tertiary"] {
        background: none !important;
        background-color: transparent !important;
        background-image: none !important;
        border: none !important;
        box-shadow: none !important;
        padding: 2px 2px !important;
        margin: 2px 0 6px 0 !important;
        text-align: left !important;
        justify-content: flex-start !important;
        transform: none !important;
        width: auto !important;
        cursor: pointer !important;
    }

    div[class*="st-key-back_step1"] button *,
    div[class*="st-key-back_step1"] button p,
    div[class*="st-key-back_step1"] button span,
    div[class*="st-key-back_step1"] button div,
    section[data-testid="stSidebar"] div[class*="st-key-back_step1"] button *,
    section[data-testid="stSidebar"] div[class*="st-key-back_step1"] button p,
    section[data-testid="stSidebar"] div[class*="st-key-back_step1"] button span,
    section[data-testid="stSidebar"] div[class*="st-key-back_step1"] button div,
    section[data-testid="stSidebar"] button[data-testid="baseButton-tertiary"] *,
    section[data-testid="stSidebar"] button[data-testid="baseButton-tertiary"] p,
    section[data-testid="stSidebar"] button[data-testid="baseButton-tertiary"] span,
    section[data-testid="stSidebar"] button[data-testid="baseButton-tertiary"] div {
        color: var(--academic-navy) !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
    }

    /* Labels de todos los widgets */
    label[data-testid="stWidgetLabel"],
    label[data-testid="stWidgetLabel"] p,
    label[data-testid="stWidgetLabel"] span {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* Sliders */
    div[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
        background-color: #00AFB5 !important;
        border: 2px solid #008B90 !important;
        box-shadow: 0 0 0 2px rgba(0, 175, 181, 0.25) !important;
    }
    div[data-testid="stSlider"] [data-baseweb="slider"] > div > div:first-child,
    div[data-testid="stSlider"] [data-baseweb="slider"] div[style*="rgb(255, 75, 75)"],
    div[data-testid="stSlider"] [data-baseweb="slider"] div[style*="#ff4b4b"] {
        background: #00AFB5 !important;
        background-color: #00AFB5 !important;
    }
    div[data-testid="stThumbValue"] {
        color: var(--tropical-teal) !important;
        font-weight: 700 !important;
        font-size: 0.88rem !important;
    }
    div[data-testid="stSliderTickBarMin"],
    div[data-testid="stSliderTickBarMax"] {
        color: var(--text-muted) !important;
        font-size: 0.78rem !important;
    }

    /* Radios y Checkboxes */
    div[data-testid="stRadio"] label span,
    div[data-testid="stRadio"] label p,
    div[data-testid="stRadio"] div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stCheckbox"] label span,
    div[data-testid="stCheckbox"] label p {
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
    }
    
    /* ========================================================================= */
    /* SELECTBOXES, DROPDOWNS E INPUTS */
    /* ========================================================================= */
    div[data-testid="stSelectbox"],
    div[data-testid="stSelectbox"] *,
    div[data-baseweb="select"],
    div[data-baseweb="select"] *,
    div[data-testid="stNumberInput"] div[data-baseweb="input"],
    div[data-testid="stNumberInput"] input,
    div[data-testid="stTextInput"] div[data-baseweb="input"],
    div[data-testid="stTextInput"] input,
    div[data-testid="stTextArea"] div[data-baseweb="textarea"],
    div[data-testid="stTextArea"] textarea {
        background-color: var(--input-bg) !important;
        background: var(--input-bg) !important;
        color: var(--text-primary) !important;
        border-color: var(--border-color) !important;
    }

    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        border: 1px solid var(--border-color) !important;
        border-radius: 6px !important;
    }

    div[data-baseweb="select"] svg,
    div[data-testid="stSelectbox"] svg {
        background-color: transparent !important;
        background: transparent !important;
        fill: var(--tropical-teal) !important;
    }

    /* Menú desplegable del Selectbox */
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] > div,
    ul[data-baseweb="menu"],
    ul[role="listbox"],
    div[data-baseweb="menu"] {
        background-color: var(--card-bg) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 6px !important;
        box-shadow: 0 4px 16px rgba(0,0,0,0.25) !important;
    }

    li[data-baseweb="menu-item"],
    li[role="option"] {
        background-color: var(--card-bg) !important;
        color: var(--text-primary) !important;
    }

    li[data-baseweb="menu-item"]:hover,
    li[role="option"]:hover,
    li[aria-selected="true"] {
        background-color: var(--badge-bg) !important;
        color: var(--tropical-teal) !important;
    }

    /* ========================================================================= */
    /* CHECKBOXES (CUADRADOS Y MARCAS) */
    /* ========================================================================= */
    div[data-testid="stCheckbox"] label > span,
    div[data-testid="stCheckbox"] label > div,
    div[data-testid="stCheckbox"] [data-baseweb="checkbox"],
    div[data-testid="stCheckbox"] [data-baseweb="checkbox"] > span,
    div[data-testid="stCheckbox"] [data-baseweb="checkbox"] > div,
    span[data-baseweb="checkbox"],
    span[data-baseweb="checkbox"] > div,
    span[data-baseweb="checkbox"] > span {
        background-color: var(--input-bg) !important;
        border: 1.5px solid var(--border-color) !important;
        border-radius: 4px !important;
        transition: all 0.2s ease !important;
    }

    div[data-testid="stCheckbox"]:hover label > span,
    div[data-testid="stCheckbox"]:hover label > div,
    div[data-testid="stCheckbox"]:hover [data-baseweb="checkbox"] {
        border-color: var(--tropical-teal) !important;
        box-shadow: 0 0 0 2px rgba(0, 175, 181, 0.25) !important;
    }

    div[data-testid="stCheckbox"] [data-baseweb="checkbox"][aria-checked="true"],
    div[data-testid="stCheckbox"] [data-baseweb="checkbox"][aria-checked="true"] *,
    div[data-testid="stCheckbox"] [aria-checked="true"],
    div[data-testid="stCheckbox"] [aria-checked="true"] > div,
    div[data-testid="stCheckbox"] [aria-checked="true"] > span,
    div[data-testid="stCheckbox"] input:checked + div,
    div[data-testid="stCheckbox"] input[type="checkbox"]:checked + div,
    span[data-baseweb="checkbox"][aria-checked="true"],
    span[data-baseweb="checkbox"][aria-checked="true"] > div {
        background-color: var(--tropical-teal) !important;
        border-color: var(--tropical-teal) !important;
    }

    div[data-testid="stCheckbox"] svg {
        background-color: transparent !important;
        fill: #FFFFFF !important;
        stroke: #FFFFFF !important;
    }

    /* Expander */
    div[data-testid="stExpander"] {
        background-color: var(--card-bg) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
    }
    div[data-testid="stExpander"] details summary,
    div[data-testid="stExpander"] details summary span,
    div[data-testid="stExpander"] details summary p {
        color: var(--academic-navy) !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
    }
    div[data-testid="stExpander"] svg {
        fill: var(--tropical-teal) !important;
    }

    /* Encabezado principal */
    .main-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding-top: 6px !important;
        padding-bottom: 8px !important;
        border-bottom: 2.5px solid var(--header-border) !important;
        margin-top: 0 !important;
        margin-bottom: 12px !important;
    }
    .main-title {
        font-family: 'STIX Two Text', 'Lora', Georgia, serif !important;
        font-size: 2.6rem !important;
        font-weight: 800 !important;
        color: var(--academic-navy) !important;
        letter-spacing: -0.5px !important;
        margin: 0 !important;
        padding-top: 4px !important;
        line-height: 1.25 !important;
    }
    .main-subtitle {
        font-family: 'Inter', sans-serif !important;
        color: var(--text-secondary) !important;
        margin: 2px 0 0 0 !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
    }
    .badge-sub {
        background-color: var(--badge-bg) !important;
        color: var(--badge-color) !important;
        border: 1px solid var(--border-dark) !important;
        padding: 6px 14px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 700;
        font-family: 'Inter', sans-serif;
        white-space: nowrap !important;
        display: inline-block !important;
    }

    /* Tablas */
    table {
        color: var(--text-primary) !important;
        background-color: var(--card-bg) !important;
    }
    table th {
        background-color: var(--tab-active-bg) !important;
        color: var(--academic-navy) !important;
        border-color: var(--border-color) !important;
    }
    table td {
        border-color: var(--border-color) !important;
        color: var(--text-secondary) !important;
    }
</style>
""".replace("__THEME_VARS__", theme_vars)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

if "engine" not in st.session_state:
    st.session_state.engine = DIPASEngine()
else:
    st.session_state.engine.__class__ = DIPASEngine
    if getattr(st.session_state.engine, "xfoil", None) is None:
        try:
            st.session_state.engine.xfoil = XFoilWrapper()
        except Exception:
            pass
    if not hasattr(st.session_state.engine, "_precision_loaded"):
        st.session_state.engine._init_surrogate()
        st.session_state.engine._precision_loaded = True

engine = st.session_state.engine

def compute_aerodynamic_diagnostic(cl_target, cd_target, reynolds, tc_target, alpha_eval, cl_sim_match, cd_sim_match, alpha_match, max_cl_val, stall_alpha, max_ld_val, cand):
    """
    Genera un diagnóstico físico formal explicando la concordancia o las causas
    de cualquier discrepancia entre los requisitos y la simulación física (Explainable AI).
    """
    err_cl_pct = ((cl_sim_match - cl_target) / max(abs(cl_target), 0.001)) * 100.0
    err_cd_pct = ((cd_sim_match - cd_target) / max(abs(cd_target), 0.001)) * 100.0
    alpha_delta = alpha_match - alpha_eval
    
    # 1. No convergencia numérica de XFOIL (Corte prematuro por burbuja laminar a bajo Reynolds)
    if max_cl_val < cl_target and stall_alpha < 6.0:
        status_color = "#EA580C" # Amber-Orange
        badge = "NO CONVERGENCIA NUMÉRICA EN XFOIL — SE RECOMIENDA SIMULACIÓN EN ANSYS FLUENT"
        explanation = f"XFOIL interrumpió el barrido polar a α = {stall_alpha:.1f}° (alcanzando CL = {max_cl_val:.2f} con pendiente dCL/dα > 0 lineal y creciente) debido a no convergencia numérica, antes de completar el rango para verificar el CL* = {cl_target:.2f} de diseño."
        cause = f"A bajo Reynolds (Re = {reynolds:,}) con espesor relativo t/c = {cand['max_tc']*100:.1f}% y curvatura trasera, se engrosa la burbuja de separación laminar (LSB) en el extradós. El método de capa límite integral 1D de XFOIL viola su hipótesis de capa límite delgada (δ ≪ c) y su algoritmo numérico de Newton-Raphson diverge."
        recommendation = "Se recomienda ejecutar la validación en <b>ANSYS Fluent (Transition SST γ-Reθ)</b> para resolver Navier-Stokes 2D con recirculación real de la burbuja, o evaluar a mayor número de Reynolds (Re ≥ 200.000)."

    # 1b. Pérdida física real alcanzada (después de un barrido completo)
    elif max_cl_val < cl_target:
        status_color = "#DC2626" # Crimson
        badge = "INSUFICIENCIA DE SUSTENTACIÓN / PÉRDIDA FÍSICA ALCANZADA"
        explanation = f"El perfil alcanza un CL máximo de {max_cl_val:.2f} (a α = {stall_alpha:.1f}°), entrando en pérdida aerodinámica antes de satisfacer el requerimiento de CL* = {cl_target:.2f}."
        cause = f"La combinación de alta sustentación requerida con espesor relativo t/c = {cand['max_tc']*100:.1f}% genera un gradiente de presión adverso excesivo en el extradós que desata el desprendimiento de la capa límite a Re = {reynolds:,}."
        recommendation = "Incrementar la curvatura media (camber), optimizar la distribución de espesor, o elevar el número de Reynolds para energizar la capa límite."
        
    # 2. Desviación de ángulo de ataque / Trim
    elif abs(alpha_delta) >= 1.0 and abs(err_cl_pct) > 6.0:
        status_color = "#D97706" # Amber
        badge = "DESVIACIÓN DE INCIDENCIA DE CALAJE (TRIM ANGLE)"
        explanation = f"El perfil satisface el CL* = {cl_target:.2f} requerido a α = {alpha_match:.1f}° en lugar de la incidencia nominal (α = {alpha_eval:.1f}°), registrando un desfase de {alpha_delta:+.1f}°."
        cause = f"La pendiente de sustentación y el ángulo de sustentación nula (α₀) están determinados por la curvatura media ({cand['max_camber']*100:.1f}%) y la distribución de espesor generada."
        recommendation = f"Fijar el ángulo de calaje del ala en α = {alpha_match:.1f}° para la misión, o incrementar la curvatura máxima si se requiere operar a menor ángulo de ataque."
        
    # 3. Penalidad de arrastre por transición / bajo Reynolds
    elif err_cd_pct > 20.0:
        status_color = "#EA580C" # Orange
        badge = "PENALIDAD DE ARRASTRE POR BURBUJA LAMINAR DE TRANSICIÓN"
        explanation = f"El coeficiente de arrastre simulado (CD = {cd_sim_match:.4f}) es un {err_cd_pct:+.1f}% mayor que el objetivo de diseño (CD* = {cd_target:.3f})."
        cause = f"A Re = {reynolds:,}, la recuperación de presión en el extradós induce la formación de una burbuja de separación laminar corta (LSB), elevando el arrastre de presión (CDp)."
        recommendation = "Aumentar ligeramente el espesor t/c para suavizar el gradiente de presión, o incorporar elementos de transición forzada (turbuladores) en el 25-35% de la cuerda."
        
    # 4. Cumplimiento Óptimo
    else:
        status_color = "#059669" # Emerald
        badge = "CONVERGENCIA FÍSICA VALIDADA"
        explanation = f"El perfil satisface los requerimientos de diseño con alta precisión (CL = {cl_sim_match:.3f} a α = {alpha_match:.1f}°, error {err_cl_pct:+.1f}%)."
        cause = f"La parametrización CST decodificada por el CVAE establece un gradiente de aceleración favorable en el extradós, situando el punto de diseño dentro del balde laminar (laminar bucket)."
        recommendation = "Geometría validada numéricamente. Apta para exportación CAD (.dat), generación de malla CFD y manufactura para ensayo experimental / UNLP."

    return {
        "color": status_color,
        "badge": badge,
        "explanation": explanation,
        "cause": cause,
        "recommendation": recommendation
    }

# Preconfiguraciones por tipo de misión
MISSION_PRESETS = {
    "Bajos Reynolds": {
        "status": "active",
        "re": 100000, "cl": 1.10, "alpha": 3.5,
        "desc": "Flujo dominado por baja inercia y capa límite laminar (Re: 100.000 — 250.000)."
    },
    "Subsónico": {
        "status": "active",
        "re": 300000, "cl": 0.85, "alpha": 3.0,
        "desc": "Régimen subsónico incompresible estándar y UAVs (Re: 200.000 — 500.000)."
    },
    "Subsónico Alto": {
        "status": "cooking",
        "re": 500000, "cl": 0.60, "alpha": 2.0,
        "desc": "Compresibilidad subsónica moderada (Mach 0.6 — 0.75). En etapa de desarrollo."
    },
    "Transónico": {
        "status": "cooking",
        "re": 500000, "cl": 0.50, "alpha": 1.5,
        "desc": "Ondas de choque y arrastre de onda transónico (Mach 0.75 — 0.95). En etapa de desarrollo."
    },
    "Supersónico": {
        "status": "cooking",
        "re": 500000, "cl": 0.35, "alpha": 1.0,
        "desc": "Flujo con ondas de choque cónicas y bordes afilados (Mach > 1.2). En etapa de desarrollo."
    },
    "Hipersónico": {
        "status": "cooking",
        "re": 500000, "cl": 0.20, "alpha": 0.5,
        "desc": "Alta entalpía y efectos termoquímicos de gas real (Mach > 5.0). En etapa de desarrollo."
    }
}

# Inicializar estados en sesión si no existen
if "sidebar_step" not in st.session_state:
    st.session_state.sidebar_step = 1
if "mission_type" not in st.session_state or st.session_state.mission_type not in MISSION_PRESETS:
    st.session_state.mission_type = "Bajos Reynolds"
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None
if "candidates" not in st.session_state:
    st.session_state.candidates = []
if "selected_cand_idx" not in st.session_state:
    st.session_state.selected_cand_idx = 0
if "last_params" not in st.session_state:
    st.session_state.last_params = {}
if "xfoil_res" not in st.session_state:
    st.session_state.xfoil_res = None

# ==============================================================================
# ENCABEZADO
# ==============================================================================
st.markdown("""
<div class="main-header">
    <div>
        <h1 class="main-title">DIPAS</h1>
        <p class="main-subtitle">
            Diseño Inverso de Perfiles Aerodinámicos mediante Autoencoders Condicionales y Simulación Multi-Fidelidad
        </p>
    </div>
    <div>
        <span class="badge-sub">UNLP</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# BARRA LATERAL: FLUJO GUIADO (PASO 1: MISIÓN Y MODELO -> PASO 2: PARÁMETROS)
# ==============================================================================
with st.sidebar:
    col_mode_t, col_mode_s = st.columns([6.8, 3.2])
    with col_mode_t:
        st.markdown("<p style='font-weight: 700; font-size: 0.82rem; margin: 4px 0 0 0; color: var(--text-primary);'>Tema Visual</p>", unsafe_allow_html=True)
    with col_mode_s:
        toggled = st.toggle("🌙", value=st.session_state.dark_mode, key="theme_toggle_btn", help="Alternar entre Modo Claro y Modo Oscuro")
        if toggled != st.session_state.dark_mode:
            st.session_state.dark_mode = toggled
            st.rerun()
    st.markdown("<div style='margin-bottom: 6px;'></div>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # PASO 1: DEFINICIÓN DE MISIÓN Y MODELO CVAE
    # --------------------------------------------------------------------------
    if st.session_state.sidebar_step == 1:
        st.markdown("### I. Tipo de Misión y Régimen")
        
        mission_options = list(MISSION_PRESETS.keys())
        current_m_idx = mission_options.index(st.session_state.mission_type) if st.session_state.mission_type in mission_options else 0
        
        def format_mission_name(m):
            if MISSION_PRESETS[m]["status"] == "cooking":
                return f"{m}  [Cooking ⏳]"
            return f"{m}  ✓"
            
        mission_type = st.selectbox(
            "Seleccionar Tipo de Misión",
            mission_options,
            index=current_m_idx,
            format_func=format_mission_name,
            help="Seleccione el régimen de vuelo y propósito aerodinámico del diseño."
        )
        st.session_state.mission_type = mission_type
        
        is_cooking = (MISSION_PRESETS[mission_type]["status"] == "cooking")
        
        # Modelo IA
        available_models = engine.get_available_cvae_models()
        model_labels = {
            "dipas_base_model.pth": "Experimento A: DIPAS Base + CFD (XFOIL → ANSYS)",
            "dipas_tl_model.pth": "Experimento B: Transfer Learning (UniFoil → ANSYS)",
            "dipas_tl_exp_model.pth": "Experimento C: Multi-Fidelidad Insignia (UniFoil → UIUC → ANSYS)"
        }
        cur_model_idx = available_models.index(st.session_state.selected_model) if st.session_state.selected_model in available_models else 0
        selected_model = st.selectbox(
            "Arquitectura CVAE / Modelo",
            available_models,
            index=cur_model_idx,
            format_func=lambda m: model_labels.get(m, m),
            help="Modelo generativo neuronal entrenado sobre coordenadas CST y CFD."
        )
        st.session_state.selected_model = selected_model
        if selected_model != engine.current_cvae_name:
            engine.load_cvae(selected_model)
            
        if is_cooking:
            st.markdown(f"""
            <div style="background-color: #FFFBEB; border: 1px solid #FDE68A; border-left: 4px solid #D97706; border-radius: 6px; padding: 10px 12px; margin: 12px 0; font-size: 0.82rem; color: #92400E;">
                ⏳ <b>Régimen en Desarrollo (Cooking):</b><br>
                El régimen <b>{mission_type}</b> está en proceso de calibración y entrenamiento numérico.<br>
                Actualmente el rango disponible en DIPAS es <b>Re = 100.000 a 500.000</b> (disponible en <i>Bajos Reynolds</i> y <i>Subsónico</i>).
            </div>
            """, unsafe_allow_html=True)
            st.button("CONTINUAR A PARÁMETROS DE DISEÑO →", use_container_width=True, disabled=True)
        else:
            model_short = model_labels.get(selected_model, selected_model).split(":")[0]
            st.markdown(f"""
            <div style="background-color: var(--badge-bg); border: 1px solid var(--border-dark); border-radius: 6px; padding: 8px 10px; margin: 12px 0; font-size: 0.82rem; color: var(--text-primary);">
                • <b>Régimen:</b> {mission_type}<br>
                • <b>Modelo:</b> {model_short}<br>
                • <b>Descripción:</b> {MISSION_PRESETS[mission_type]["desc"]}
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("CONTINUAR A PARÁMETROS DE DISEÑO →", use_container_width=True):
                preset_p1 = MISSION_PRESETS[st.session_state.mission_type]
                st.session_state.val_cl = float(preset_p1["cl"])
                st.session_state.val_re = int(preset_p1["re"])
                st.session_state.val_alpha = float(preset_p1["alpha"])
                st.session_state.sidebar_step = 2
                st.rerun()

    # --------------------------------------------------------------------------
    # PASO 2: PARÁMETROS AERODINÁMICOS Y SÍNTESIS INVERSA
    # --------------------------------------------------------------------------
    else:
        # Resumen del paso 1 con opción de retroceso
        model_labels = {
            "dipas_base_model.pth": "Experimento A: DIPAS Base",
            "dipas_tl_model.pth": "Experimento B: Transfer Learning",
            "dipas_tl_exp_model.pth": "Experimento C: Multi-Fidelidad Insignia"
        }
        cur_model = st.session_state.selected_model or engine.current_cvae_name
        model_name_disp = model_labels.get(cur_model, cur_model)
        
        st.markdown(f"""
        <div style="background-color: var(--badge-bg); border-left: 3px solid var(--tropical-teal); border-radius: 4px; padding: 6px 10px; margin-bottom: 6px; font-size: 0.80rem; color: var(--text-primary);">
            <b>Misión:</b> {st.session_state.mission_type}<br>
            <b>Modelo CVAE:</b> {model_name_disp}
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("← Cambiar misión o modelo", key="back_step1", type="tertiary"):
            st.session_state.sidebar_step = 1
            st.rerun()
            
        st.markdown("---")
        st.markdown("### II. Parámetros Aerodinámicos de Diseño")
        
        preset = MISSION_PRESETS[st.session_state.mission_type]
        
        # Inicialización de estado sincronizado si no existen
        if "val_cl" not in st.session_state:
            st.session_state.val_cl = float(preset["cl"])
        if "val_re" not in st.session_state:
            st.session_state.val_re = int(preset["re"])
        if "val_alpha" not in st.session_state:
            st.session_state.val_alpha = float(preset["alpha"])

        # ----------------------------------------------------------------------
        # 1. Coeficiente de Sustentación Objetivo (CL*)
        # ----------------------------------------------------------------------
        c1_head, c1_mode = st.columns([8.5, 1.5])
        with c1_head:
            st.markdown("<p style='font-weight: 600; font-size: 0.88rem; margin: 0; color: var(--text-primary);'>Sustentación (CL*)</p>", unsafe_allow_html=True)
        with c1_mode:
            manual_cl = st.checkbox("Manual CL", value=False, key="chk_manual_cl", label_visibility="collapsed", help="Activar caja para escribir valor exacto")
            
        if manual_cl:
            cl_target = st.number_input(
                "Sustentación (CL*)",
                min_value=0.20, max_value=1.80,
                value=float(st.session_state.val_cl),
                step=0.01, format="%.2f",
                label_visibility="collapsed",
                help="Rango sugerido: 0.20 a 1.80"
            )
        else:
            cl_target = st.slider(
                "Sustentación (CL*)",
                min_value=0.20, max_value=1.80,
                value=float(st.session_state.val_cl),
                step=0.05,
                label_visibility="collapsed",
                help="Sustentación requerida para la condición de crucero o maniobra."
            )
        st.session_state.val_cl = cl_target

        st.markdown("<div style='margin-top: 4px;'></div>", unsafe_allow_html=True)

        # ----------------------------------------------------------------------
        # 2. Número de Reynolds (Re)
        # ----------------------------------------------------------------------
        c2_head, c2_mode = st.columns([8.5, 1.5])
        with c2_head:
            st.markdown("<p style='font-weight: 600; font-size: 0.88rem; margin: 0; color: var(--text-primary);'>Reynolds (Re)</p>", unsafe_allow_html=True)
        with c2_mode:
            manual_re = st.checkbox("Manual RE", value=False, key="chk_manual_re", label_visibility="collapsed", help="Activar caja para escribir valor exacto")
            
        if manual_re:
            reynolds = st.number_input(
                "Reynolds (Re)",
                min_value=100000, max_value=500000,
                value=int(st.session_state.val_re),
                step=5000,
                label_visibility="collapsed",
                help="Rango sugerido: 100.000 a 500.000"
            )
        else:
            re_options = [100000, 150000, 200000, 250000, 300000, 400000, 500000]
            closest_re = min(re_options, key=lambda x: abs(x - st.session_state.val_re))
            reynolds = st.select_slider(
                "Reynolds (Re)",
                options=re_options,
                value=closest_re,
                format_func=lambda x: f"{x:,}".replace(",", "."),
                label_visibility="collapsed"
            )
        st.session_state.val_re = int(reynolds)

        st.markdown("<div style='margin-top: 4px;'></div>", unsafe_allow_html=True)

        # ----------------------------------------------------------------------
        # 3. Ángulo de Ataque de Crucero (alpha)
        # ----------------------------------------------------------------------
        c3_head, c3_mode = st.columns([8.5, 1.5])
        with c3_head:
            st.markdown("<p style='font-weight: 600; font-size: 0.88rem; margin: 0; color: var(--text-primary);'>Incidencia (α)</p>", unsafe_allow_html=True)
        with c3_mode:
            manual_alpha = st.checkbox("Manual Alpha", value=False, key="chk_manual_alpha", label_visibility="collapsed", help="Activar caja para escribir valor exacto")
            
        if manual_alpha:
            alpha_eval = st.number_input(
                "Incidencia (α)",
                min_value=-2.0, max_value=10.0,
                value=float(st.session_state.val_alpha),
                step=0.1, format="%.1f",
                label_visibility="collapsed",
                help="Rango sugerido: -2.0° a 10.0°"
            )
        else:
            alpha_eval = st.slider(
                "Incidencia (α)",
                min_value=-2.0, max_value=10.0,
                value=float(st.session_state.val_alpha),
                step=0.5, format="%.1f°",
                label_visibility="collapsed",
                help="Incidencia nominal en la que se evalúa la sustentación."
            )
        st.session_state.val_alpha = float(alpha_eval)

        selected_model = st.session_state.selected_model or engine.current_cvae_name
        n_samples = 500  # Evaluación masiva de 500 variantes latentes
        seed_val = 42
        
        st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
        generate_btn = st.button("GENERAR PERFILES ÓPTIMOS", use_container_width=True)

# Parámetros por defecto para inicialización si se está en Paso 1 y aún no se definieron
if st.session_state.sidebar_step == 1:
    preset = MISSION_PRESETS[st.session_state.mission_type]
    cl_target = float(preset["cl"])
    reynolds = int(preset["re"])
    alpha_eval = float(preset["alpha"])
    seed_val = 42
    selected_model = st.session_state.selected_model or engine.current_cvae_name
    n_samples = 500
    generate_btn = False
else:
    selected_model = st.session_state.selected_model or engine.current_cvae_name

# Sincronización de parámetros
current_params = {
    "cl": cl_target, "re": reynolds, "alpha": alpha_eval, 
    "n_samples": n_samples, "model": selected_model
}
params_changed = (st.session_state.last_params != current_params)

# Manejo de generación con prioridad absoluta en CL* y física
if generate_btn or (params_changed and st.session_state.sidebar_step == 2) or len(st.session_state.candidates) == 0:
    with st.spinner(f"Sintetizando perfiles óptimos para CL* = {cl_target:.2f} a α = {alpha_eval:.1f}°..."):
        candidates = engine.generate_airfoils(
            cl_target=cl_target,
            cd_target=0.015,
            reynolds=reynolds,
            tc_target=0.12,
            n_samples=n_samples,
            seed=seed_val,
            eval_alpha=alpha_eval
        )
        st.session_state.candidates = candidates
        st.session_state.selected_cand_idx = 0
        st.session_state.last_params = current_params
        st.session_state.xfoil_res = None

candidates = st.session_state.candidates
cand_idx = st.session_state.selected_cand_idx
if cand_idx >= len(candidates):
    cand_idx = 0
active_cand = candidates[cand_idx]

# ==============================================================================
# PESTAÑAS PRINCIPALES DEL SISTEMA (ACADEMIC LABELS)
# ==============================================================================
tab_geom, tab_valid, tab_export, tab_about = st.tabs([
    "1. Síntesis Geométrica y Variantes",
    "2. Validación Numérica (XFOIL / ANSYS)",
    "3. Exportación CAD y Manufactura",
    "4. Acerca de DIPAS"
])

# ------------------------------------------------------------------------------
# TAB 1: GEOMETRÍA Y TOP CANDIDATOS
# ------------------------------------------------------------------------------
with tab_geom:
    top5 = candidates[:5]
    
    col_plot, col_ranking = st.columns([6.2, 3.8])
    
    with col_ranking:
        st.markdown("""
        <div class="dipas-card">
            <h4 style="color: var(--academic-navy); font-family: 'STIX Two Text', serif; margin-top: 0; font-size: 1.05rem;">
                Arquetipos de Diseño Especializados
            </h4>
            <p style="font-size: 0.80rem; color: var(--text-secondary); margin-bottom: 2px;">
                5 variantes físicas optimizadas sobre el manifold latente CVAE.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        options = [
            f"#{c['rank']} {c.get('archetype_tag', 'Variante')} (L/D = {c['surrogate_ld']:.1f}, t/c = {c['max_tc']*100:.1f}%)" 
            for c in top5
        ]
        
        sel_idx = st.selectbox(
            "Seleccionar Geometría Activa:",
            range(len(top5)),
            format_func=lambda i: options[i],
            index=cand_idx if cand_idx < len(top5) else 0
        )
        
        if sel_idx != st.session_state.selected_cand_idx:
            st.session_state.selected_cand_idx = sel_idx
            st.session_state.xfoil_res = None
            st.rerun()

        # Descripción del arquetipo seleccionado
        cur_cand = top5[sel_idx]
        st.markdown(f"""
        <div style='margin-top: 6px; margin-bottom: 8px; padding: 7px 11px; border-radius: 6px; background-color: var(--badge-bg); border-left: 3px solid var(--tropical-teal); font-size: 0.78rem; color: var(--text-secondary); line-height: 1.35;'>
            <b style='color: var(--text-primary);'>{cur_cand.get('archetype_name', 'Variante')}:</b><br>
            {cur_cand.get('archetype_desc', '')}
        </div>
        """, unsafe_allow_html=True)

        # Checkbox para superponer en el gráfico grande
        show_all = st.checkbox("Superponer 5 Variantes en Gráfico Principal", value=False)
        
        # Tabla resumen compacta con variables de tema
        top5_rows_list = []
        for idx, c in enumerate(top5):
            bg_col = 'var(--badge-bg)' if idx == sel_idx else 'transparent'
            arch_label = c.get('archetype_tag', f"Var #{c['rank']}")
            top5_rows_list.append(
                f"<tr style='background-color: {bg_col}; border-bottom: 1px solid var(--border-color);'>"
                f"<td style='padding: 6px 6px; font-weight: 700; color: var(--academic-navy);'>#{c['rank']}</td>"
                f"<td style='padding: 6px 6px; text-align: left; font-size: 0.75rem; color: var(--text-primary);'>{arch_label}</td>"
                f"<td style='padding: 6px 6px; color: var(--text-primary);'>{c['surrogate_cl']:.3f}</td>"
                f"<td style='padding: 6px 6px; color: var(--text-primary);'>{c['surrogate_cd']:.4f}</td>"
                f"<td style='padding: 6px 6px; font-weight: 600; color: var(--tropical-teal);'>{c['surrogate_ld']:.1f}</td>"
                f"<td style='padding: 6px 6px; color: var(--text-secondary);'>{c['max_tc']*100:.1f}%</td>"
                f"</tr>"
            )
        top5_rows = "".join(top5_rows_list)
        
        table_top5_html = f"""
        <div style='overflow-x: auto; border: 1px solid var(--border-color); border-radius: 8px; margin-top: 6px; background-color: var(--card-bg);'>
            <table style='width: 100%; border-collapse: collapse; font-family: Inter, sans-serif; font-size: 0.80rem; text-align: center;'>
                <thead>
                    <tr style='background-color: var(--tab-active-bg); border-bottom: 2px solid var(--border-color); color: var(--academic-navy);'>
                        <th style='padding: 6px 6px;'>#</th>
                        <th style='padding: 6px 6px; text-align: left;'>Enfoque</th>
                        <th style='padding: 6px 6px;'>CL</th>
                        <th style='padding: 6px 6px;'>CD</th>
                        <th style='padding: 6px 6px;'>L/D</th>
                        <th style='padding: 6px 6px;'>t/c</th>
                    </tr>
                </thead>
                <tbody>
                    {top5_rows}
                </tbody>
            </table>
        </div>
        """
        st.markdown(table_top5_html, unsafe_allow_html=True)
        
    with col_plot:
        # Configuración de paleta para Plotly
        if is_dark:
            p_paper = "#18314F"
            p_plot = "#0D0630"
            p_text = "#FFFFFF"
            p_grid = "#384E77"
            p_zero = "#384E77"
            p_airfoil_line = "#00AFB5"
            p_airfoil_fill = "rgba(0, 175, 181, 0.22)"
            p_camber_line = "#94A3B8"
        else:
            p_paper = "#FFFFFF"
            p_plot = "#F8FAFC"
            p_text = "#0F172A"
            p_grid = "#E2E8F0"
            p_zero = "#CBD5E1"
            p_airfoil_line = "#0F2C59"
            p_airfoil_fill = "rgba(15, 44, 89, 0.08)"
            p_camber_line = "#64748B"

        fig = go.Figure()
        
        if show_all:
            colors = ["#00AFB5", "#0284C7", "#059669", "#D97706", "#A855F7"] if is_dark else ["#0F2C59", "#0284C7", "#059669", "#D97706", "#7C3AED"]
            for idx, c in enumerate(top5):
                xc = np.concatenate([np.flip(c["x"]), c["x"]])
                yc = np.concatenate([np.flip(c["y_upper"]), c["y_lower"]])
                is_selected = (idx == sel_idx)
                
                fig.add_trace(go.Scatter(
                    x=xc, y=yc,
                    mode='lines',
                    fill='toself' if is_selected else None,
                    fillcolor=p_airfoil_fill if is_selected else None,
                    line=dict(
                        color=colors[idx % len(colors)],
                        width=3.0 if is_selected else 1.8,
                        dash='solid' if is_selected else 'dash'
                    ),
                    name=f"#{c['rank']} {c.get('archetype_tag', 'Variante')} (L/D={c['surrogate_ld']:.1f}, t/c={c['max_tc']*100:.1f}%)" + (" [Activo]" if is_selected else ""),
                    hoverinfo='x+y'
                ))
            plot_title = f"Superposición de Arquetipos de Diseño Decodificados (Re = {reynolds:,})"
        else:
            x = np.array(active_cand["x"])
            y_u = np.array(active_cand["y_upper"])
            y_l = np.array(active_cand["y_lower"])
            camber = np.array(active_cand["camber"])
            
            # Superficie de perfil (área rellena)
            x_closed = np.concatenate([np.flip(x), x])
            y_closed = np.concatenate([np.flip(y_u), y_l])
            
            fig.add_trace(go.Scatter(
                x=x_closed, y=y_closed,
                fill='toself',
                fillcolor=p_airfoil_fill,
                line=dict(color=p_airfoil_line, width=2.8),
                name=f"#{active_cand['rank']} {active_cand.get('archetype_name', 'Perfil Activo')}",
                hoverinfo='x+y'
            ))
            
            # Línea media de curvatura (Camber line)
            fig.add_trace(go.Scatter(
                x=x, y=camber,
                mode='lines',
                line=dict(color=p_camber_line, width=1.5, dash='dash'),
                name='Línea Media (Camber)',
                hoverinfo='x+y'
            ))
            
            # Marcador de Espesor Máximo
            max_idx = int(np.argmax(y_u - y_l))
            fig.add_trace(go.Scatter(
                x=[x[max_idx], x[max_idx]],
                y=[y_l[max_idx], y_u[max_idx]],
                mode='lines+markers',
                line=dict(color='#D97706', width=2),
                marker=dict(size=6, color='#D97706'),
                name=f"t/c máx = {active_cand['max_tc']*100:.1f}%",
                hoverinfo='text',
                text=[f"t/c = {active_cand['max_tc']*100:.2f}% en x/c = {x[max_idx]:.2f}"]*2
            ))
            plot_title = f"Geometría Decodificada — Variante #{active_cand['rank']} (t/c = {active_cand['max_tc']*100:.1f}%, Re = {reynolds:,})"
        
        fig.update_layout(
            title=dict(
                text=plot_title,
                font=dict(family='STIX Two Text, serif', color=p_text, size=14),
                x=0.01,
                y=0.98,
                xanchor='left'
            ),
            paper_bgcolor=p_paper,
            plot_bgcolor=p_plot,
            xaxis=dict(
                title=dict(
                    text="<b>x / c</b>",
                    font=dict(family='Inter, sans-serif', color=p_text, size=12),
                    standoff=6
                ),
                tickfont=dict(family='Inter, sans-serif', color=p_text, size=10),
                linecolor='#94A3B8',
                range=[-0.05, 1.05],
                gridcolor=p_grid,
                zerolinecolor=p_zero,
                constrain='domain'
            ),
            yaxis=dict(
                title=dict(text="<b>y / c</b>", font=dict(family='Inter, sans-serif', color=p_text, size=12)),
                tickfont=dict(family='Inter, sans-serif', color=p_text, size=10),
                linecolor='#94A3B8',
                range=[-0.20, 0.20],
                gridcolor=p_grid,
                zerolinecolor=p_zero,
                scaleanchor="x",
                scaleratio=1
            ),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.28,
                xanchor="center",
                x=0.5,
                font=dict(family='Inter, sans-serif', color=p_text, size=10)
            ),
            margin=dict(l=35, r=15, t=30, b=75),
            height=310
        )
        
        st.plotly_chart(fig, theme=None, use_container_width=True)
        
        # Fila de métricas KPI
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            cl_val = f"{active_cand['surrogate_cl']:.3f}" if active_cand['surrogate_cl'] is not None else "N/A"
            st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-title">CL Estimado</div>
                <div class="kpi-value">{cl_val}</div>
                <div class="kpi-sub">Target: {cl_target:.2f} (α={alpha_eval:.1f}°)</div>
            </div>
            """, unsafe_allow_html=True)
        with k2:
            cd_val = f"{active_cand['surrogate_cd']:.4f}" if active_cand['surrogate_cd'] is not None else "N/A"
            st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-title">CD Estimado</div>
                <div class="kpi-value">{cd_val}</div>
                <div class="kpi-sub">Arrastre a {alpha_eval:.1f}°</div>
            </div>
            """, unsafe_allow_html=True)
        with k3:
            ld_val = f"{active_cand['surrogate_ld']:.1f}" if active_cand['surrogate_ld'] else "N/A"
            st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-title">Eficiencia L/D</div>
                <div class="kpi-value">{ld_val}</div>
                <div class="kpi-sub">Surrogate AI</div>
            </div>
            """, unsafe_allow_html=True)
        with k4:
            st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-title">Espesor t/c</div>
                <div class="kpi-value">{active_cand['max_tc']*100:.1f}%</div>
                <div class="kpi-sub">x/c = {active_cand['x_tc_max']:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        with k5:
            cm_val = active_cand.get('surrogate_cm', float(-2.0 * active_cand['max_camber'] * (1.0 - active_cand['x_camber_max'])))
            st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-title">CM Estimado (c/4)</div>
                <div class="kpi-value">{cm_val:.4f}</div>
                <div class="kpi-sub">Trimado a {alpha_eval:.1f}°</div>
            </div>
            """, unsafe_allow_html=True)
            
        # Feedback instantáneo de la IA en Pestaña 1
        cl_est = active_cand['surrogate_cl'] if active_cand['surrogate_cl'] is not None else 0.0
        cd_est = active_cand['surrogate_cd'] if active_cand['surrogate_cd'] is not None else 0.0
        tc_act = active_cand['max_tc']
        err_cl = abs(cl_est - cl_target) / max(cl_target, 0.01) * 100.0
        
        if active_cand['is_valid'] and err_cl < 10.0:
            f_color = "#059669"
            f_badge = "SÍNTESIS GENERATIVA FÍSICAMENTE COHERENTE"
            f_msg = f"El perfil decodificado <b>Variante #{active_cand['rank']}</b> alcanza una sustentación estimada de <b>CL = {cl_est:.3f}</b> a α = {alpha_eval:.1f}° (t/c = {tc_act*100:.1f}%, CM = {cm_val:.4f}) con eficiencia L/D = {active_cand['surrogate_ld']:.1f}."
            f_hint = "Proceda a la pestaña <b>'2. Validación Numérica'</b> y ejecute la simulación XFOIL para verificar la capa límite y obtener la ficha de polares completa."
        else:
            f_color = "#D97706"
            f_badge = "VARIACIÓN DETECTADA EN ESPACIO LATENTE"
            f_msg = f"La <b>Variante #{active_cand['rank']}</b> presenta un t/c de {tc_act*100:.1f}% y CL estimado de {cl_est:.3f} (desviación del objetivo nominal)."
            f_hint = "Puede seleccionar cualquiera de las otras variantes en el panel lateral o evaluar la polar en la Pestaña 2."
            
        st.markdown(f"""
        <div style="background-color: var(--card-bg); border: 1px solid var(--border-color); border-left: 4px solid {f_color}; border-radius: 8px; padding: 12px 16px; margin-top: 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
            <div style="font-family: 'STIX Two Text', serif; font-weight: 700; font-size: 0.90rem; color: {f_color}; margin-bottom: 4px;">[{f_badge}]</div>
            <div style="font-size: 0.84rem; color: var(--text-primary); margin-bottom: 4px;">{f_msg}</div>
            <div style="font-size: 0.82rem; color: var(--tropical-teal);">{f_hint}</div>
        </div>
        """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# TAB 2: VALIDACIÓN NUMÉRICA (XFOIL / ANSYS FLUENT)
# ------------------------------------------------------------------------------
with tab_valid:
    st.markdown("""
    <div class="dipas-card-accent">
        <h3 style="color: var(--academic-navy); font-family: 'STIX Two Text', serif; margin: 0 0 4px 0; font-size: 1.25rem;">Módulo de Validación y Simulación Aerodinámica</h3>
        <p style="color: #475569; margin: 0; font-size: 0.88rem;">
            Verificación aerodinámica de la geometría decodificada mediante métodos de capa límite acoplada (XFOIL) y CFD de alta fidelidad (ANSYS Fluent Transition SST).
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    val_mode = st.radio(
        "Seleccionar Motor de Validación:",
        ["XFOIL (Solución Numérica Viscosa Rápida)", "ANSYS Fluent (CFD Alta Fidelidad - Modelo Transition SST)"],
        horizontal=True
    )
    
    # ------------------ MODO XFOIL ------------------
    if "XFOIL" in val_mode:
        col_ctrl, col_res = st.columns([3, 7])
        
        with col_ctrl:
            st.markdown("<h4 style='color: #0F2C59; font-family: STIX Two Text, serif;'>Parámetros de Barrido</h4>", unsafe_allow_html=True)
            a_start = st.number_input("Incidencia Inicial (α min)", value=-4.0, step=1.0)
            a_end = st.number_input("Incidencia Final (α max)", value=14.0, step=1.0)
            a_step = st.number_input("Paso Angular (Δα)", value=1.0, step=0.5)
            
            run_xfoil_btn = st.button("EJECUTAR SIMULACIÓN XFOIL", use_container_width=True)
            
            if run_xfoil_btn:
                with st.spinner("Ejecutando simulación aerodinámica multi-fidelidad..."):
                    try:
                        res = engine.run_xfoil_validation(
                            active_cand,
                            reynolds=reynolds,
                            alpha_start=a_start,
                            alpha_end=a_end,
                            alpha_step=a_step,
                            eval_alpha=alpha_eval
                        )
                        st.session_state.xfoil_res = res
                    except Exception as e:
                        st.error(f"Error en la ejecución: {e}")
                        
        with col_res:
            xf_res = st.session_state.xfoil_res
            if xf_res and xf_res.get("polar") is not None:
                is_fb = xf_res.get("is_fallback", False)
                solver_label = "Surrogate Multi-Fidelidad (Red Tensorial RANS/XFOIL)" if is_fb else "XFOIL 6.99 (Método de Paneles Viscosos eⁿ — Mark Drela, MIT)"
                st.success(f"✅ **Simulación Convergida con Éxito** • **Motor Activo:** `{solver_label}`")
                
                diag = xf_res.get("diagnostic_info")
                if diag:
                    with st.expander("🔍 Ver Diagnóstico de Ejecución del Solver", expanded=False):
                        st.code(diag)
                    
                p_df = pd.DataFrame(xf_res["polar"])
                
                # Valores estimados del perfil decodificado
                tc_target = active_cand['max_tc']
                cd_target = active_cand['surrogate_cd'] if active_cand['surrogate_cd'] is not None else 0.015
                
                # Extraer valores clave para la comparación
                # 1. Punto más cercano al CL objetivo
                cl_diffs = np.abs(p_df["CL"] - cl_target)
                closest_idx = int(np.argmin(cl_diffs))
                alpha_match = p_df["alpha"].iloc[closest_idx]
                cl_sim_match = p_df["CL"].iloc[closest_idx]
                cd_sim_match = p_df["CD"].iloc[closest_idx]
                ld_sim_match = cl_sim_match / max(cd_sim_match, 0.0001)
                
                # 2. Máxima eficiencia L/D y CL máximo
                p_df["L_D"] = p_df["CL"] / np.clip(p_df["CD"], 0.0001, None)
                max_ld_idx = int(p_df["L_D"].idxmax())
                max_ld_val = p_df["L_D"].iloc[max_ld_idx]
                max_ld_alpha = p_df["alpha"].iloc[max_ld_idx]
                
                max_cl_idx = int(p_df["CL"].idxmax())
                max_cl_val = p_df["CL"].iloc[max_cl_idx]
                stall_alpha = p_df["alpha"].iloc[max_cl_idx]
                
                # Errores porcentuales
                err_cl_pct = ((cl_sim_match - cl_target) / max(abs(cl_target), 0.001)) * 100.0
                err_cd_pct = ((cd_sim_match - cd_target) / max(abs(cd_target), 0.001)) * 100.0
                
                # Tarjetas KPI de Comparación
                st.markdown("<h4 style='color: #0F2C59; font-family: STIX Two Text, serif; margin-bottom: 8px;'>Comparación de Rendimiento: Objetivo vs IA vs Simulación</h4>", unsafe_allow_html=True)
                c_k1, c_k2, c_k3, c_k4 = st.columns(4)
                
                with c_k1:
                    st.markdown(f"""
                    <div class="kpi-container">
                        <div class="kpi-title">Sustentación CL</div>
                        <div class="kpi-value">{cl_sim_match:.3f}</div>
                        <div class="kpi-sub">Target: {cl_target:.2f} (Δ {err_cl_pct:+.1f}%)</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with c_k2:
                    st.markdown(f"""
                    <div class="kpi-container">
                        <div class="kpi-title">Arrastre CD (α={alpha_match:.1f}°)</div>
                        <div class="kpi-value">{cd_sim_match:.4f}</div>
                        <div class="kpi-sub">IA: {cd_target:.4f} (Δ {err_cd_pct:+.1f}%)</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with c_k3:
                    ai_ld = active_cand['surrogate_ld']
                    ai_ld_str = f"{ai_ld:.1f}" if ai_ld else "N/A"
                    st.markdown(f"""
                    <div class="kpi-container">
                        <div class="kpi-title">Eficiencia (L/D) máx</div>
                        <div class="kpi-value">{max_ld_val:.1f}</div>
                        <div class="kpi-sub">a α={max_ld_alpha:.1f}° (IA: {ai_ld_str})</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with c_k4:
                    st.markdown(f"""
                    <div class="kpi-container">
                        <div class="kpi-title">CL Máximo (Stall)</div>
                        <div class="kpi-value">{max_cl_val:.2f}</div>
                        <div class="kpi-sub">Pérdida en α = {stall_alpha:.1f}°</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                # Diagnóstico Físico Inteligente (Explainable AI)
                diag = compute_aerodynamic_diagnostic(
                    cl_target=cl_target,
                    cd_target=cd_target,
                    reynolds=reynolds,
                    tc_target=tc_target,
                    alpha_eval=alpha_eval,
                    cl_sim_match=cl_sim_match,
                    cd_sim_match=cd_sim_match,
                    alpha_match=alpha_match,
                    max_cl_val=max_cl_val,
                    stall_alpha=stall_alpha,
                    max_ld_val=max_ld_val,
                    cand=active_cand
                )
                
                st.markdown(f"""
                <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-left: 4px solid {diag['color']}; border-radius: 8px; padding: 14px 18px; margin: 14px 0 18px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="font-family: 'STIX Two Text', serif; font-weight: 700; font-size: 0.95rem; color: {diag['color']};">[{diag['badge']}]</span>
                        <span style="font-size: 0.75rem; color: #475569; background: #F1F5F9; padding: 2px 10px; border-radius: 6px; border: 1px solid #CBD5E1; font-family: Inter, sans-serif;">Diagnóstico Físico DIPAS</span>
                    </div>
                    <div style="font-size: 0.86rem; color: #0F172A; margin-bottom: 5px;">
                        <b>• Evaluación:</b> {diag['explanation']}
                    </div>
                    <div style="font-size: 0.86rem; color: #334155; margin-bottom: 5px;">
                        <b>• Causa Física:</b> {diag['cause']}
                    </div>
                    <div style="font-size: 0.86rem; color: #0284C7;">
                        <b>• Recomendación de Ingeniería:</b> {diag['recommendation']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Gráficos polares interactivos con líneas de referencia de objetivos (Academic Light)
                fig_pol = make_subplots(
                    rows=1, cols=3,
                    subplot_titles=(
                        "<b>CL vs α (Sustentación)</b>",
                        "<b>CD vs α (Arrastre)</b>",
                        "<b>CL / CD vs α (Eficiencia)</b>"
                    )
                )
                
                # 1. CL vs alpha + Línea Target
                fig_pol.add_trace(go.Scatter(
                    x=p_df["alpha"], y=p_df["CL"],
                    mode='lines+markers', line=dict(color='#0F2C59', width=2.5),
                    marker=dict(size=4, color='#0F2C59'), name="XFOIL CL"
                ), row=1, col=1)
                
                fig_pol.add_trace(go.Scatter(
                    x=[p_df["alpha"].min(), p_df["alpha"].max()],
                    y=[cl_target, cl_target],
                    mode='lines',
                    line=dict(color='#D97706', width=2, dash='dash'),
                    name=f"CL* Objetivo ({cl_target:.2f})"
                ), row=1, col=1)
                
                # 2. CD vs alpha + Línea Target
                fig_pol.add_trace(go.Scatter(
                    x=p_df["alpha"], y=p_df["CD"],
                    mode='lines+markers', line=dict(color='#0284C7', width=2.5),
                    marker=dict(size=4, color='#0284C7'), name="XFOIL CD"
                ), row=1, col=2)
                
                fig_pol.add_trace(go.Scatter(
                    x=[p_df["alpha"].min(), p_df["alpha"].max()],
                    y=[cd_target, cd_target],
                    mode='lines',
                    line=dict(color='#059669', width=2, dash='dash'),
                    name=f"CD* Objetivo ({cd_target:.3f})"
                ), row=1, col=2)
                
                # 3. L/D vs alpha
                fig_pol.add_trace(go.Scatter(
                    x=p_df["alpha"], y=p_df["L_D"],
                    mode='lines+markers', line=dict(color='#059669', width=2.5),
                    marker=dict(size=4, color='#059669'), name="XFOIL L/D"
                ), row=1, col=3)
                
                fig_pol.update_layout(
                    paper_bgcolor=p_paper,
                    plot_bgcolor=p_plot,
                    height=340,
                    margin=dict(l=25, r=25, t=45, b=85),
                    legend=dict(
                        orientation="h",
                        yanchor="top",
                        y=-0.30,
                        xanchor="center",
                        x=0.5,
                        font=dict(family='Inter, sans-serif', color=p_text, size=10)
                    )
                )
                for i in range(1, 4):
                    fig_pol.update_xaxes(
                        gridcolor=p_grid, zerolinecolor=p_zero,
                        title=dict(text="<b>α (grados)</b>", font=dict(family='Inter, sans-serif', color=p_text, size=11)),
                        tickfont=dict(family='Inter, sans-serif', color=p_text, size=10),
                        linecolor='#94A3B8',
                        row=1, col=i
                    )
                    fig_pol.update_yaxes(
                        gridcolor=p_grid, zerolinecolor=p_zero,
                        tickfont=dict(family='Inter, sans-serif', color=p_text, size=10),
                        linecolor='#94A3B8',
                        row=1, col=i
                    )
                    
                st.plotly_chart(fig_pol, theme=None, use_container_width=True)
                
                # Gráficos inferiores: Polar de Arrastre (CL vs CD) + Distribución de Presiones Cp
                col_drag_pol, col_cp_plot = st.columns(2)
                
                with col_drag_pol:
                    fig_drag = go.Figure()
                    fig_drag.add_trace(go.Scatter(
                        x=p_df["CD"], y=p_df["CL"],
                        mode='lines+markers',
                        line=dict(color='#00AFB5' if is_dark else '#0F2C59', width=2.5),
                        marker=dict(size=4, color='#00AFB5' if is_dark else '#0F2C59'),
                        name="Polar XFOIL"
                    ))
                    # Punto objetivo
                    fig_drag.add_trace(go.Scatter(
                        x=[cd_target], y=[cl_target],
                        mode='markers',
                        marker=dict(size=12, color='#D97706', symbol='star'),
                        name=f"Punto de Diseño ({cd_target:.3f}, {cl_target:.2f})"
                    ))
                    fig_drag.update_layout(
                        title=dict(
                            text="<b>Polar de Arrastre (CL vs CD)</b>",
                            font=dict(family='STIX Two Text, serif', color=p_text, size=14)
                        ),
                        paper_bgcolor=p_paper,
                        plot_bgcolor=p_plot,
                        xaxis=dict(
                            title=dict(text="<b>CD (Arrastre)</b>", font=dict(family='Inter, sans-serif', color=p_text, size=12)),
                            tickfont=dict(family='Inter, sans-serif', color=p_text, size=10),
                            linecolor='#94A3B8',
                            gridcolor=p_grid,
                            zerolinecolor=p_zero
                        ),
                        yaxis=dict(
                            title=dict(text="<b>CL (Sustentación)</b>", font=dict(family='Inter, sans-serif', color=p_text, size=12)),
                            tickfont=dict(family='Inter, sans-serif', color=p_text, size=10),
                            linecolor='#94A3B8',
                            gridcolor=p_grid,
                            zerolinecolor=p_zero
                        ),
                        height=310,
                        margin=dict(l=25, r=25, t=45, b=80),
                        legend=dict(font=dict(family='Inter, sans-serif', color=p_text, size=10), orientation="h", yanchor="top", y=-0.32, x=0.5, xanchor="center")
                    )
                    st.plotly_chart(fig_drag, theme=None, use_container_width=True)
                    
                with col_cp_plot:
                    if xf_res.get("cp") and isinstance(xf_res["cp"], dict) and "x" in xf_res["cp"] and len(xf_res["cp"]["x"]) > 0:
                        cp_df = pd.DataFrame(xf_res["cp"])
                        fig_cp = go.Figure()
                        fig_cp.add_trace(go.Scatter(
                            x=cp_df["x"], y=cp_df["Cp"],
                            mode='lines+markers',
                            line=dict(color='#00AFB5' if is_dark else '#0284C7', width=2),
                            marker=dict(size=3),
                            name=f"Cp a α = {xf_res['evaluated_alpha_cp']}°"
                        ))
                        fig_cp.update_layout(
                            title=dict(
                                text=f"<b>Distribución de Presión Cp a α = {xf_res['evaluated_alpha_cp']}°</b>",
                                font=dict(family='STIX Two Text, serif', color=p_text, size=14)
                            ),
                            paper_bgcolor=p_paper,
                            plot_bgcolor=p_plot,
                            xaxis=dict(
                                title=dict(text="<b>x / c</b>", font=dict(family='Inter, sans-serif', color=p_text, size=12)),
                                tickfont=dict(family='Inter, sans-serif', color=p_text, size=10),
                                linecolor='#94A3B8',
                                gridcolor=p_grid,
                                zerolinecolor=p_zero
                            ),
                            yaxis=dict(
                                title=dict(text="<b>- Cp</b>", font=dict(family='Inter, sans-serif', color=p_text, size=12)),
                                tickfont=dict(family='Inter, sans-serif', color=p_text, size=10),
                                linecolor='#94A3B8',
                                autorange="reversed",
                                gridcolor=p_grid,
                                zerolinecolor=p_zero
                            ),
                            height=310,
                            margin=dict(l=25, r=25, t=45, b=80),
                            legend=dict(font=dict(family='Inter, sans-serif', color=p_text, size=10), orientation="h", yanchor="top", y=-0.32, x=0.5, xanchor="center")
                        )
                        st.plotly_chart(fig_cp, theme=None, use_container_width=True)
                        
                # --------------------------------------------------------------
                # FICHA DE PARÁMETROS GLOBALES DE LA POLAR AERODINÁMICA
                # --------------------------------------------------------------
                st.markdown("<hr style='margin: 16px 0 12px 0;'>", unsafe_allow_html=True)
                st.markdown(f"""
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                    <h4 style="color: #0F2C59; font-family: 'STIX Two Text', serif; margin: 0; font-size: 1.15rem;">
                        Parámetros Globales de la Polar Aerodinámica
                    </h4>
                    <span style="font-size: 0.76rem; color: #475569; background: #EEF2F6; padding: 3px 10px; border-radius: 4px; border: 1px solid #CBD5E1;">
                        Simulación XFOIL • Re = {reynolds:,}
                    </span>
                </div>
                """, unsafe_allow_html=True)
                
                # Cálculos de Parámetros Globales
                min_cd_idx = int(p_df["CD"].idxmin())
                min_cd_val = float(p_df["CD"].iloc[min_cd_idx])
                min_cd_alpha = float(p_df["alpha"].iloc[min_cd_idx])
                
                # Ángulo de sustentación nula alpha_0
                if (p_df["CL"].min() <= 0.05 and p_df["CL"].max() >= -0.05):
                    try:
                        lin_mask = (p_df["alpha"] <= 6.0) & (p_df["alpha"] >= -6.0)
                        p_lin = p_df[lin_mask].sort_values("CL")
                        alpha_0 = float(np.interp(0.0, p_lin["CL"], p_lin["alpha"]))
                    except Exception:
                        alpha_0 = float(np.interp(0.0, p_df["CL"], p_df["alpha"]))
                else:
                    p_lin = p_df[p_df["alpha"] <= 4.0]
                    if len(p_lin) >= 2:
                        slope, intercept = np.polyfit(p_lin["alpha"], p_lin["CL"], 1)
                        alpha_0 = float(-intercept / max(slope, 0.001))
                    else:
                        alpha_0 = 0.0
                        
                # Pendiente de sustentación CL,alpha (en 1/° y 1/rad)
                lin_zone = p_df[(p_df["alpha"] >= max(alpha_0 - 1.0, -4.0)) & (p_df["alpha"] <= min(alpha_0 + 6.0, 6.0))]
                if len(lin_zone) >= 2:
                    p_fit = np.polyfit(lin_zone["alpha"], lin_zone["CL"], 1)
                    cl_alpha_deg = float(p_fit[0])
                    cl_alpha_rad = float(cl_alpha_deg * (180.0 / np.pi))
                else:
                    cl_alpha_deg = 0.105
                    cl_alpha_rad = 6.01
                    
                # CM0 (Momento a sustentación nula) y CM en crucero
                if "CM" in p_df.columns:
                    try:
                        cm_0_val = float(np.interp(alpha_0, p_df["alpha"], p_df["CM"]))
                        cm_cruise_val = float(p_df["CM"].iloc[closest_idx])
                    except Exception:
                        cm_0_val = active_cand.get("surrogate_cm", -0.05)
                        cm_cruise_val = cm_0_val
                else:
                    cm_0_val = active_cand.get("surrogate_cm", -0.05)
                    cm_cruise_val = cm_0_val

                cl_at_max_ld = float(p_df["CL"].iloc[max_ld_idx])
                cd_at_max_ld = float(p_df["CD"].iloc[max_ld_idx])

                polar_rows = [
                    ("Sustentación Máxima", "<i>C</i><sub>L,max</sub>", f"<b>{max_cl_val:.3f}</b>", f"en α = {stall_alpha:.1f}° (Pérdida / Stall)"),
                    ("Ángulo de Sustentación Nula", "<i>α</i><sub>0</sub>", f"<b>{alpha_0:.2f}°</b>", "<i>C</i><sub>L</sub>(α₀) = 0 (Curvatura media)"),
                    ("Pendiente de Sustentación", "<i>C</i><sub>L,α</sub>", f"<b>{cl_alpha_deg:.4f} /°</b> &nbsp;<span style='color:#64748B;'>({cl_alpha_rad:.2f} /rad)</span>", "Régimen lineal de sustentación"),
                    ("Arrastre Mínimo de Perfil", "<i>C</i><sub>D,min</sub>", f"<b>{min_cd_val:.4f}</b>", f"en α = {min_cd_alpha:.1f}° (Fondo balde laminar)"),
                    ("Máxima Eficiencia Aerodinámica", "(<i>L/D</i>)<sub>max</sub>", f"<b>{max_ld_val:.1f}</b>", f"en α = {max_ld_alpha:.1f}° (Planeo óptimo)"),
                    ("Sustentación en Máx. Eficiencia", "<i>C</i><sub>L | (L/D)max</sub>", f"<b>{cl_at_max_ld:.3f}</b>", "Sustentación en crucero óptimo"),
                    ("Arrastre en Máx. Eficiencia", "<i>C</i><sub>D | (L/D)max</sub>", f"<b>{cd_at_max_ld:.4f}</b>", "Arrastre en crucero de máxima eficiencia"),
                    ("Momento a Sustentación Nula", "<i>C</i><sub>m0</sub>", f"<b>{cm_0_val:.4f}</b>", "Momento intrínseco respecto a c/4"),
                    ("Momento en Punto de Diseño", "<i>C</i><sub>m (c/4)</sub>", f"<b>{cm_cruise_val:.4f}</b>", f"a α = {alpha_match:.1f}° (Condición de trimado)")
                ]

                rows_html = "".join([
                    f"<tr style='background-color: {'var(--card-bg)' if i % 2 == 0 else 'var(--badge-bg)'}; border-bottom: 1px solid var(--border-color);'>"
                    f"<td style='padding: 9px 14px; color: var(--text-primary); font-weight: 500;'>{p_name}</td>"
                    f"<td style='padding: 9px 14px; text-align: center; color: var(--academic-navy); font-family: STIX Two Text, serif; font-size: 0.95rem;'>{p_sym}</td>"
                    f"<td style='padding: 9px 14px; text-align: center; color: var(--tropical-teal); font-family: STIX Two Text, serif; font-size: 0.95rem;'>{p_val}</td>"
                    f"<td style='padding: 9px 14px; color: var(--text-secondary); font-size: 0.80rem;'>{p_cond}</td>"
                    f"</tr>"
                    for i, (p_name, p_sym, p_val, p_cond) in enumerate(polar_rows)
                ])

                table_html = (
                    "<div style='overflow-x: auto; border: 1px solid var(--border-color); border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.03); margin-top: 6px;'>"
                    "<table style='width: 100%; border-collapse: collapse; font-family: Inter, sans-serif; font-size: 0.84rem; text-align: left;'>"
                    "<thead><tr style='background-color: var(--tab-active-bg); border-bottom: 2px solid var(--border-color); color: var(--academic-navy);'>"
                    "<th style='padding: 10px 14px; font-weight: 700;'>PARÁMETRO AERODINÁMICO</th>"
                    "<th style='padding: 10px 14px; font-weight: 700; text-align: center;'>SÍMBOLO</th>"
                    "<th style='padding: 10px 14px; font-weight: 700; text-align: center;'>VALOR NUMÉRICO</th>"
                    "<th style='padding: 10px 14px; font-weight: 700;'>CONDICIÓN / RÉGIMEN</th>"
                    "</tr></thead>"
                    f"<tbody>{rows_html}</tbody>"
                    "</table></div>"
                )
                st.markdown(table_html, unsafe_allow_html=True)
            else:
                st.info("Presione **'EJECUTAR SIMULACIÓN XFOIL'** para calcular las polares aerodinámicas y verificar los requerimientos.")

    # ------------------ MODO ANSYS FLUENT ------------------
    else:
        is_ansys_installed, fluent_exe, ansys_version = engine.detect_ansys_fluent()
        
        col_ans_info, col_ans_actions = st.columns([5, 5])
        
        with col_ans_info:
            if is_ansys_installed:
                st.markdown(f"""
                <div class="dipas-card-accent" style="border-left-color: #059669;">
                    <h4 style="color: #059669; font-family: 'STIX Two Text', serif; margin: 0 0 6px 0; font-size: 1.15rem;">ANSYS Fluent Detectado</h4>
                    <p style="color: var(--text-primary); font-size: 0.88rem; margin: 0;">
                        <b>Versión:</b> {ansys_version}<br>
                        <b>Ruta Ejecutable:</b> <code>{fluent_exe}</code>
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="dipas-card">
                    <h4 style="color: #D97706; font-family: 'STIX Two Text', serif; margin: 0 0 6px 0; font-size: 1.15rem;">Instalación Local de ANSYS no Detectada</h4>
                    <p style="color: #475569; font-size: 0.88rem; margin: 0;">
                        No se detectó un ejecutable de Fluent en los directorios predeterminados de Windows.
                        Puede <b>exportar las geometrías y coordenadas</b> para mallado en cualquier estación de cálculo externa.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
        with col_ans_actions:
            st.markdown("<h4 style='color: #0F2C59; font-family: STIX Two Text, serif;'>Acciones CFD</h4>", unsafe_allow_html=True)
            if is_ansys_installed:
                if st.button("ABRIR GUI DE ANSYS FLUENT LOCAL", use_container_width=True):
                    with st.spinner(f"Generando malla 2D y preparando caso en ANSYS Fluent (Re = {reynolds:,}, α = {alpha_eval:.1f}°)..."):
                        try:
                            engine.launch_ansys_gui(active_cand, reynolds=reynolds, alpha=alpha_eval)
                            st.success(f"✓ ANSYS Fluent iniciado. Malla 2D (`current_airfoil.msh`) lista en `data/`.")
                        except Exception as e:
                            st.error(f"Error al iniciar Fluent: {e}")
            else:
                st.button("ABRIR GUI DE ANSYS FLUENT LOCAL", disabled=True, use_container_width=True)
                
            # Lectura / Generación de malla para descarga
            msh_path = engine.data_dir / "current_airfoil.msh"
            msh_bytes = b""
            if not msh_path.exists():
                try:
                    from fluent_mesh_generator import generate_airfoil_mesh
                    chord = 0.200
                    x = np.array(active_cand["x"])
                    y_u = np.array(active_cand["y_upper"])
                    y_l = np.array(active_cand["y_lower"])
                    x_coords = np.concatenate([np.flip(x), x[1:]]) * chord
                    y_coords = np.concatenate([np.flip(y_u), y_l[1:]]) * chord
                    coords_xy = list(zip(x_coords, y_coords))
                    generate_airfoil_mesh(coords_xy, str(msh_path), chord=chord)
                except Exception:
                    pass
            if msh_path.exists():
                with open(msh_path, "rb") as f:
                    msh_bytes = f.read()

            st.download_button(
                label="📥 DESCARGAR MALLA FLUENT 2D (.MSH)",
                data=msh_bytes,
                file_name=f"malla_dipas_var_{active_cand['rank']}.msh",
                mime="application/octet-stream",
                use_container_width=True
            )
                
            export_dat_btn = st.button("EXPORTAR ARCHIVO .DAT PARA SPACECLAIM / ICEM CFD", use_container_width=True)
            if export_dat_btn:
                target_dat = engine.data_dir / f"airfoil_top_{active_cand['rank']}.dat"
                engine.export_to_selig_format(active_cand, str(target_dat))
                st.success(f"Archivo exportado en: `{target_dat}`")
                
        # Card explicativa de carga en Fluent
        msh_path_disp = str(engine.data_dir / "current_airfoil.msh")
        jou_path_disp = str(engine.data_dir / "setup_fluent_case.jou")
        st.markdown(f"""
        <div style="background-color: var(--card-bg); border: 1px solid var(--border-color); border-radius: 8px; padding: 12px 16px; margin-top: 14px;">
            <h5 style="color: var(--academic-navy); font-family: 'STIX Two Text', serif; margin: 0 0 6px 0; font-size: 1.05rem;">
                📌 Instrucciones de Carga en ANSYS Fluent
            </h5>
            <div style="font-size: 0.83rem; color: var(--text-primary); line-height: 1.5;">
                <ol style="margin: 6px 0 6px 0; padding-left: 20px;">
                    <li><b>Con el botón de descarga directa:</b> Descarga la malla <code>.msh</code> con el botón azul y ábrela en cualquier equipo o estación CFD con <code>File → Read → Mesh...</code></li>
                    <li><b>En ejecución local:</b> Abre Fluent con el botón superior y selecciona directamente el archivo generado:
                        <br><code style="background: var(--badge-bg); padding: 2px 6px; border-radius: 4px; color: var(--tropical-teal);">{msh_path_disp}</code>
                    </li>
                </ol>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# TAB 3: EXPORTACIÓN CAD Y MANUFACTURA
# ------------------------------------------------------------------------------
with tab_export:
    col_exp_cfg, col_exp_view = st.columns([4, 6])
    
    with col_exp_cfg:
        st.markdown("""
        <div class="dipas-card">
            <h4 style="color: var(--academic-navy); font-family: 'STIX Two Text', serif; margin-top: 0; font-size: 1.15rem;">Parámetros Dimensionales y Fabricación</h4>
            <p style="font-size: 0.85rem; color: var(--text-secondary);">Escalado milimétrico para manufactura aditiva FDM, corte CNC y validación geométrica.</p>
        </div>
        """, unsafe_allow_html=True)
        
        chord_mm = st.number_input("Cuerda Real del Modelo (mm)", value=200.0, step=10.0, min_value=50.0, max_value=1000.0)
        profile_name = st.text_input("Designación Oficial del Perfil", value=f"DIPAS_VAR_{active_cand['rank']}")
        
        dat_content = engine.export_to_selig_format(active_cand, chord_mm=chord_mm, name=profile_name)
        csv_content = engine.export_to_csv(active_cand, chord_mm=chord_mm)
        
        # Malla binaria para exportación
        msh_exp_bytes = b""
        msh_exp_path = engine.data_dir / "current_airfoil.msh"
        if msh_exp_path.exists():
            with open(msh_exp_path, "rb") as f:
                msh_exp_bytes = f.read()
        
        st.download_button(
            label="DESCARGAR COORDENADAS (.DAT SELIG / XFOIL)",
            data=dat_content,
            file_name=f"{profile_name.lower()}.dat",
            mime="text/plain",
            use_container_width=True
        )
        
        st.download_button(
            label="DESCARGAR TABLA DE COORDENADAS (.CSV)",
            data=csv_content,
            file_name=f"{profile_name.lower()}_coords.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.download_button(
            label="📥 DESCARGAR MALLA CFD 2D (.MSH FLUENT)",
            data=msh_exp_bytes,
            file_name=f"{profile_name.lower()}_mesh.msh",
            mime="application/octet-stream",
            use_container_width=True
        )
        
    with col_exp_view:
        st.markdown("<h4 style='color: var(--academic-navy); font-family: STIX Two Text, serif;'>Vista Previa del Archivo de Coordenadas (.DAT Selig / UIUC)</h4>", unsafe_allow_html=True)
        st.text_area("Contenido .dat", dat_content, height=350)

# ------------------------------------------------------------------------------
# TAB 4: ACERCA DE DIPAS (OBJETIVO, RÉGIMEN Y DATASETS / REFERENCIAS)
# ------------------------------------------------------------------------------
with tab_about:
    st.markdown("""
    <div class="dipas-card-accent" style="margin-bottom: 16px;">
        <h3 style="color: var(--academic-navy); font-family: 'STIX Two Text', serif; margin: 0 0 6px 0; font-size: 1.30rem;">
            DIPAS • Plataforma de Diseño Inverso Aerodinámico
        </h3>
        <p style="color: var(--text-primary); font-size: 0.90rem; line-height: 1.5; margin: 0 0 8px 0;">
            <b>DIPAS</b> (<i>Diseño Inverso para Perfiles mediante Autoencoders y Simulación</i>) es una plataforma computacional avanzada de síntesis generativa aerodinámica impulsada por redes neuronales <b>CVAE</b> (<i>Conditional Variational Autoencoders</i>) acopladas a modelos sustitutos multi-fidelidad (<b>XFOIL & CFD RANS</b>).
        </p>
        <p style="color: var(--text-secondary); font-size: 0.88rem; line-height: 1.5; margin: 0;">
            Su objetivo fundamental es <b>automatizar y acelerar el diseño inverso y la selección del perfil aerodinámico óptimo</b> para satisfacer de forma simultánea y rigurosa los requerimientos de sustentación (<i>C</i><sub>L</sub><sup>*</sup>), arrastre (<i>C</i><sub>D</sub><sup>*</sup>) y ángulo de incidencia (<i>α</i>).
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col_reg, col_lic = st.columns([4.5, 5.5])
    
    with col_reg:
        st.markdown("""
        <div class="dipas-card" style="height: 100%;">
            <h4 style="color: var(--academic-navy); font-family: 'STIX Two Text', serif; margin: 0 0 8px 0; font-size: 1.10rem;">
                🎯 Régimen Operativo y Aplicación
            </h4>
            <div style="font-size: 0.85rem; color: var(--text-primary); line-height: 1.5;">
                <p>
                    Actualmente, los modelos CVAE y sustitutos de DIPAS están entrenados, validados y calibrados específicamente para el régimen de <b>Bajos Números de Reynolds (<i>Re</i> = 100.000 – 500.000)</b>.
                </p>
                <p>
                    Este régimen físico de capa límite laminar y fenómenos de transición es fundamental en el diseño de:
                </p>
                <ul style="padding-left: 18px; margin-bottom: 0; color: var(--text-secondary);">
                    <li><b>Vehículos Aéreos No Tripulados (UAVs / Drones tácticos)</b>.</li>
                    <li><b>Planeadores de alto rendimiento y motoveleros</b>.</li>
                    <li><b>Aviación ultraligera y micro-aeronaves eléctricas</b>.</li>
                    <li><b>Microaerogeneradores y palas eólicas de baja velocidad</b>.</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_lic:
        st.markdown("""
        <div class="dipas-card" style="height: 100%;">
            <h4 style="color: var(--academic-navy); font-family: 'STIX Two Text', serif; margin: 0 0 8px 0; font-size: 1.10rem;">
                🔬 Datasets de Elaboración Propia (DIPAS Framework)
            </h4>
            <div style="font-size: 0.83rem; color: var(--text-primary); line-height: 1.45;">
                <div style="margin-bottom: 12px; background: var(--badge-bg); padding: 8px 10px; border-radius: 6px; border-left: 3px solid var(--tropical-teal);">
                    <b style="color: var(--academic-navy);">1. Dataset DIPAS Base (Capa Límite Acoplada / XFOIL):</b><br>
                    <span style="color: var(--text-secondary);">
                        • <b>Perfiles únicos generados:</b> 9.931 geometrías analíticas CST de orden 4.<br>
                        • <b>Simulaciones numéricas viscosas:</b> 146.916 evaluaciones aerodinámicas con barrido de polar (<i>α</i> = -4.0° a +10.0°) en <i>Re</i> = 100.000 a 300.000.
                    </span>
                </div>
                <div style="background: var(--badge-bg); padding: 8px 10px; border-radius: 6px; border-left: 3px solid var(--accent-emerald);">
                    <b style="color: var(--academic-navy);">2. Dataset DIPAS CFD High-Fidelity (RANS Transition SST):</b><br>
                    <span style="color: var(--text-secondary);">
                        • <b>Perfiles de alta fidelidad:</b> 300 geometrías representativas del espacio latente.<br>
                        • <b>Simulaciones CFD 2D resueltas:</b> 10.806 cálculos numéricos en ANSYS Fluent mediante modelo de cuatro ecuaciones <i>γ-Re<sub>θt</sub></i> SST (resolución de burbuja de separación laminar LSB) en <i>Re</i> = 100.000 a 350.000 y <i>α</i> = -2° a +8°.
                    </span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
    
    col_ref_data, col_ref_soft = st.columns([5, 5])
    
    with col_ref_data:
        st.markdown("""
        <div class="dipas-card" style="height: 100%;">
            <h4 style="color: var(--academic-navy); font-family: 'STIX Two Text', serif; margin: 0 0 8px 0; font-size: 1.05rem;">
                📚 Datasets de Referencia y Transfer Learning
            </h4>
            <div style="font-size: 0.82rem; color: var(--text-secondary); line-height: 1.45;">
                <div style="margin-bottom: 10px;">
                    • <b style="color: var(--text-primary);">UniFoil Geometric Dataset:</b><br>
                    10.000 geometrías de perfiles aerodinámicos parametrizados mediante coeficientes CST para el pre-entrenamiento global de la red CVAE generativa.
                </div>
                <div style="margin-bottom: 10px;">
                    • <b style="color: var(--text-primary);">UIUC Airfoil Coordinates Database:</b><br>
                    1.620 perfiles experimentales analizados y ajustados bajo representación CST (Prof. Michael Selig et al., University of Illinois at Urbana-Champaign).
                </div>
                <div>
                    • <b style="color: var(--text-primary);">UIUC Low-Speed Wind Tunnel Dataset:</b><br>
                    6.560 puntos de ensayo aerodinámico experimental en túnel de viento a bajos números de Reynolds sobre 66 geometrías de perfiles clásicos y modernos.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_ref_soft:
        st.markdown("""
        <div class="dipas-card" style="height: 100%;">
            <h4 style="color: var(--academic-navy); font-family: 'STIX Two Text', serif; margin: 0 0 8px 0; font-size: 1.05rem;">
                🏛️ Métodos, Software y Licencias
            </h4>
            <div style="font-size: 0.82rem; color: var(--text-secondary); line-height: 1.45;">
                <div style="margin-bottom: 10px;">
                    • <b style="color: var(--text-primary);">XFOIL Subsonic Airfoil Analysis System:</b><br>
                    Prof. Mark Drela (MIT). Código abierto distribuido bajo licencia GNU General Public License (GPL).
                </div>
                <div style="margin-bottom: 10px;">
                    • <b style="color: var(--text-primary);">Parametrización Geométrica CST (Class-Shape Transformation):</b><br>
                    Brenda M. Kulfan (Boeing). Método analítico para definición suave y continua de superficies aerodinámicas.
                </div>
                <div>
                    • <b style="color: var(--text-primary);">Gmsh Mesh Generator:</b><br>
                    Christophe Geuzaine & Jean-François Remacle. Generador de mallas bidimensionales no estructuradas con capa límite prismática (GNU GPL).
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; margin-top: 24px; padding-top: 14px; border-top: 1px solid var(--border-color); font-size: 0.82rem; color: var(--text-muted);">
        <b>Сделано Jota</b> (<i>Hecho por Jota</i>) • Universidad Nacional de La Plata (UNLP)
    </div>
    """, unsafe_allow_html=True)

