import streamlit as st
st.set_page_config(page_title="Data Agent Console", layout="wide", page_icon="🔬")
import streamlit.components.v1 as components
import json
from pymongo import MongoClient,ReadPreference
from neo4j import GraphDatabase
import time
from bson import SON
import polars as pl
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

import math

@st.cache_resource
def get_mongo_client():
    try:
        client = MongoClient(
            "mongodb://mongo-primary:27017/?replicaSet=rs0",
            serverSelectionTimeoutMS=3000
        )
        client.admin.command("ping")
        return client, True
    except Exception as e:
        return None, False

@st.cache_resource
def get_neo4j_driver():
    for attempt in range(10):
        try:
            driver = GraphDatabase.driver(
                "bolt://neo4j:7687",
                auth=("neo4j", "password"),
                connection_timeout=5
            )
            with driver.session() as session:
                session.run("RETURN 1").single()
            return driver, True
        except Exception as e:
            if attempt == 9:
                return None, False
            time.sleep(2)
    return None, False

@st.cache_data(ttl=30)
def fetch_pipeline_metrics(_client):
    db = _client["connectome"]
    
    collections = ["raw_connections", "clean_connections", "gold_analytics"]
    metrics = []
    
    for col_name in collections:
        col = db[col_name]
        count = col.estimated_document_count()
        
        stats = db.command(SON([
            ("collStats", col_name),
            ("scale", 1024 * 1024)
        ]))
        
        metrics.append({
            "layer": col_name.replace("_connections", "").replace("_analytics", ""),
            "documents": count,
            "storage_mb": round(stats.get("storageSize", 0), 2),
            "avg_doc_kb": round(stats.get("avgObjSize", 0) / 1024, 2) if stats.get("avgObjSize") else 0
        })
    
    return pl.DataFrame(metrics)

@st.cache_data(ttl=60)
def analyze_processing_quality(_client):
    db = _client["connectome"]
    
    raw_sample = list(db["raw_connections"].aggregate([
        {"$sample": {"size": 1000}},
        {"$project": {
            "field_count": {"$size": {"$objectToArray": "$$ROOT"}},
            "has_entities": {"$cond": [{"$ifNull": ["$entities", False]}, 1, 0]},
            "has_weight": {"$cond": [{"$ifNull": ["$weight", False]}, 1, 0]}
        }}
    ]))
    
    clean_sample = list(db["clean_connections"].aggregate([
        {"$sample": {"size": 1000}},
        {"$project": {
            "field_count": {"$size": {"$objectToArray": "$$ROOT"}},
            "has_entities": {"$cond": [{"$ifNull": ["$entities", False]}, 1, 0]},
            "has_weight": {"$cond": [{"$ifNull": ["$weight", False]}, 1, 0]},
            "validated": {"$cond": [{"$ifNull": ["$validation_status", False]}, 1, 0]}
        }}
    ]))
    
    df_raw = pl.DataFrame(raw_sample)
    df_clean = pl.DataFrame(clean_sample)
    
    return {
        "raw_avg_fields": df_raw["field_count"].mean(),
        "clean_avg_fields": df_clean["field_count"].mean(),
        "raw_entity_coverage": df_raw["has_entities"].mean() * 100,
        "clean_entity_coverage": df_clean["has_entities"].mean() * 100,
        "validation_rate": df_clean.get_column("validated").mean() * 100 if "validated" in df_clean.columns else 0
    }
def render_system_overview(mongo_ok, neo4j_ok):
    st.title("Data Agent Console")
    
    col1, col2, col3, col4 = st.columns(4)
    
    client, _ = get_mongo_client()
    
    if mongo_ok and client:
        db = client["connectome"]
        
        raw_count = db["raw_connections"].estimated_document_count()
        clean_count = db["clean_connections"].estimated_document_count()
        
        with col1:
            st.metric("Raw Connections", f"{raw_count/1e6:.2f}M", delta="Ingested")
        
        with col2:
            retention = (clean_count / raw_count * 100) if raw_count > 0 else 0
            st.metric("Clean Connections", f"{clean_count/1e6:.2f}M", delta=f"{retention:.1f}%")
        
        with col3:
            try:
                region_count = db["gold_region_agg"].estimated_document_count()
                st.metric("Regions", f"{region_count:,}", delta="Aggregated")
            except:
                st.metric("Regions", "—", delta="Pending")
        
        with col4:
            try:
                tensor_count = db["gold_tensor"].estimated_document_count()
                st.metric("Tensors", f"{tensor_count:,}", delta="Generated")
            except:
                st.metric("Tensors", "—", delta="Pending")
    else:
        with col1:
            st.metric("MongoDB", "OFFLINE")
        with col2:
            st.metric("Status", "Disconnected")
    
    st.divider()
    
    if not mongo_ok or not client:
        st.error("MongoDB unavailable. Check replica set.")
        return
    
    try:
        db = client["connectome"]
        
        tab1, tab2, tab3 = st.tabs(["🔥 Live Metrics", "📊 Pipeline Flow", "⚡ Performance"])
        
        with tab1:
            collections_data = []
            for col_name in ["raw_connections", "clean_connections", "gold_region_agg", "gold_tensor"]:
                try:
                    col = db[col_name]
                    count = col.estimated_document_count()
                    
                    stats = db.command(SON([("collStats", col_name), ("scale", 1024 * 1024)]))
                    storage = stats.get("storageSize", 0)
                    
                    collections_data.append({
                        "collection": col_name,
                        "count": count,
                        "storage_mb": storage
                    })
                except:
                    pass
            
            if collections_data:
                df = pl.DataFrame(collections_data)
                
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=df["collection"].to_list(),
                    y=df["count"].to_list(),
                    text=[f"{c/1e6:.2f}M" if c > 1e6 else f"{c:,}" for c in df["count"].to_list()],
                    textposition='outside',
                    marker=dict(
                        color=df["count"].to_list(),
                        colorscale='Viridis',
                        showscale=False,
                        line=dict(color='white', width=2)
                    ),
                    hovertemplate='<b>%{x}</b><br>Documents: %{y:,}<br>Storage: %{customdata:.2f} MB<extra></extra>',
                    customdata=df["storage_mb"].to_list()
                ))
                
                fig.update_layout(
                    height=500,
                    plot_bgcolor='#0e1117',
                    paper_bgcolor='#0e1117',
                    font=dict(color='white', size=14),
                    xaxis=dict(
                        showgrid=False,
                        color='white',
                        tickfont=dict(size=12)
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridcolor='#333',
                        color='white',
                        type='log'
                    ),
                    margin=dict(t=20, b=80, l=60, r=20)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    total_docs = df["count"].sum()
                    st.metric("Total Documents", f"{total_docs/1e6:.2f}M")
                
                with col2:
                    total_storage = df["storage_mb"].sum()
                    st.metric("Total Storage", f"{total_storage:.1f} MB")
                
                with col3:
                    avg_size = (total_storage / total_docs * 1024) if total_docs > 0 else 0
                    st.metric("Avg Size", f"{avg_size:.2f} KB")
        
        with tab2:
            pipeline = [
                {"$limit": 100},
                {"$group": {
                    "_id": "$groups.neuropil",
                    "connections": {"$sum": 1},
                    "synapses": {"$sum": "$counts.syn_count"}
                }},
                {"$sort": {"synapses": -1}},
                {"$limit": 15}
            ]
            
            results = list(db["clean_connections"].aggregate(pipeline, allowDiskUse=True))
            
            if results:
                regions = [r["_id"] for r in results if r["_id"]]
                connections = [r["connections"] for r in results if r["_id"]]
                synapses = [r["synapses"] for r in results if r["_id"]]
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=synapses,
                    y=connections,
                    mode='markers+text',
                    marker=dict(
                        size=[s/100 for s in synapses],
                        color=synapses,
                        colorscale='Plasma',
                        showscale=True,
                        colorbar=dict(
                            title=dict(text="Synapses", font=dict(color='white')),
                            tickfont=dict(color='white')
                        ),
                        line=dict(color='white', width=2)
                    ),
                    text=regions,
                    textposition="top center",
                    textfont=dict(size=10, color='white'),
                    hovertemplate='<b>%{text}</b><br>Connections: %{y:,}<br>Synapses: %{x:,}<extra></extra>'
                ))
                
                fig.update_layout(
                    title=dict(text="Region Activity Map", font=dict(size=18, color='white')),
                    xaxis_title="Total Synapses",
                    yaxis_title="Connections",
                    height=500,
                    plot_bgcolor='#0e1117',
                    paper_bgcolor='#0e1117',
                    font=dict(color='white'),
                    xaxis=dict(gridcolor='#333', color='white'),
                    yaxis=dict(gridcolor='#333', color='white')
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            metrics_pipeline = [
                {"$match": {
                    "$or": [
                        {"metrics.gaba_avg": {"$gt": 0}},
                        {"metrics.glut_avg": {"$gt": 0}}
                    ]
                }},
                {"$limit": 1000},
                {"$project": {
                    "gaba": "$metrics.gaba_avg",
                    "glut": "$metrics.glut_avg",
                    "ach": "$metrics.ach_avg"
                }}
            ]
            
            results = list(db["clean_connections"].aggregate(metrics_pipeline, allowDiskUse=True))
            
            if results:
                df = pl.DataFrame(results)
                
                for col_name in ['gaba', 'glut', 'ach']:
                    if col_name in df.columns:
                        df = df.with_columns(pl.col(col_name).cast(pl.Float64, strict=False).fill_null(0))
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=df["gaba"].to_list() if "gaba" in df.columns else [],
                    y=df["glut"].to_list() if "glut" in df.columns else [],
                    mode='markers',
                    marker=dict(
                        size=4,
                        color=df["ach"].to_list() if "ach" in df.columns else [],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(
                            title=dict(text="ACh", font=dict(color='white')),
                            tickfont=dict(color='white')
                        ),
                        opacity=0.6
                    ),
                    hovertemplate='GABA: %{x:.4f}<br>Glutamate: %{y:.4f}<extra></extra>'
                ))
                
                fig.update_layout(
                    title=dict(text="GABA vs Glutamate Distribution", font=dict(size=18, color='white')),
                    xaxis_title="GABA Concentration",
                    yaxis_title="Glutamate Concentration",
                    height=500,
                    plot_bgcolor='#0e1117',
                    paper_bgcolor='#0e1117',
                    font=dict(color='white'),
                    xaxis=dict(gridcolor='#333', color='white'),
                    yaxis=dict(gridcolor='#333', color='white')
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    gaba_mean = df["gaba"].mean() if "gaba" in df.columns else 0
                    st.metric("GABA", f"{gaba_mean:.4f}")
                
                with col2:
                    glut_mean = df["glut"].mean() if "glut" in df.columns else 0
                    st.metric("Glutamate", f"{glut_mean:.4f}")
                
                with col3:
                    ach_mean = df["ach"].mean() if "ach" in df.columns else 0
                    st.metric("Acetylcholine", f"{ach_mean:.4f}")
    
    except Exception as e:
        st.error(f"Error loading metrics: {e}")
        st.exception(e)
@st.cache_data(ttl=60)
def analyze_raw_data_distribution(_client):
    db = _client["connectome"]
    col = db["raw_connections"]
    
    pipeline = [
        {"$sample": {"size": 5000}},
        {"$project": {
            "field_count": {"$size": {"$objectToArray": "$$ROOT"}},
            "has_source": {"$cond": [{"$ifNull": ["$source", False]}, 1, 0]},
            "has_target": {"$cond": [{"$ifNull": ["$target", False]}, 1, 0]},
            "has_weight": {"$cond": [{"$ifNull": ["$weight", False]}, 1, 0]},
            "has_timestamp": {"$cond": [{"$ifNull": ["$timestamp", False]}, 1, 0]},
        }}
    ]
    
    results = list(col.aggregate(pipeline, allowDiskUse=True))
    
    if not results:
        return None
    
    df = pl.DataFrame(results)
    
    return {
        "field_stats": df.select([
            pl.col("field_count").mean().alias("avg_fields"),
            pl.col("field_count").min().alias("min_fields"),
            pl.col("field_count").max().alias("max_fields"),
            pl.col("field_count").std().alias("std_fields")
        ]),
        "coverage": df.select([
            (pl.col("has_source").mean() * 100).alias("source_pct"),
            (pl.col("has_target").mean() * 100).alias("target_pct"),
            (pl.col("has_weight").mean() * 100).alias("weight_pct"),
            (pl.col("has_timestamp").mean() * 100).alias("timestamp_pct")
        ]),
        "field_distribution": df["field_count"].to_list()
    }

@st.cache_data(ttl=60)
def get_field_type_analysis(_client):
    db = _client["connectome"]
    col = db["raw_connections"]
    
    sample_size = 1000
    samples = list(col.find({}).limit(sample_size))
    
    if not samples:
        return None
    
    field_types = {}
    field_nulls = {}
    
    for doc in samples:
        for key, value in doc.items():
            if key == "_id":
                continue
            
            if key not in field_types:
                field_types[key] = {}
                field_nulls[key] = 0
            
            type_name = type(value).__name__
            field_types[key][type_name] = field_types[key].get(type_name, 0) + 1
            
            if value is None:
                field_nulls[key] += 1
    
    analysis = []
    for field, types in field_types.items():
        dominant_type = max(types.items(), key=lambda x: x[1])
        null_rate = (field_nulls[field] / sample_size) * 100
        
        analysis.append({
            "field": field,
            "dominant_type": dominant_type[0],
            "type_consistency": f"{(dominant_type[1] / sample_size * 100):.1f}%",
            "null_rate": f"{null_rate:.1f}%",
            "unique_types": len(types)
        })
    
    return pl.DataFrame(analysis)

@st.cache_data(ttl=60)
def get_paginated_raw_data(_client, skip=0, limit=50):
    db = _client["connectome"]
    col = db["raw_connections"]
    
    docs = list(col.find({}, {"_id": 0}).skip(skip).limit(limit))
    
    return docs
def analyze_clean_connectome_data(_client):
    db = _client["connectome"]
    col = db["clean_connections"]
    
    pipeline = [
        {"$match": {
            "$or": [
                {"metrics.gaba_avg": {"$gt": 0}},
                {"metrics.glut_avg": {"$gt": 0}},
                {"metrics.ach_avg": {"$gt": 0}}
            ]
        }},
        {"$limit": 10000},
        {"$project": {
            "neuropil": "$groups.neuropil",
            "syn_count": "$counts.syn_count",
            "gaba_avg": "$metrics.gaba_avg",
            "ach_avg": "$metrics.ach_avg",
            "glut_avg": "$metrics.glut_avg",
            "oct_avg": "$metrics.oct_avg",
            "ser_avg": "$metrics.ser_avg",
            "da_avg": "$metrics.da_avg"
        }}
    ]
    
    results = list(col.aggregate(pipeline, allowDiskUse=True))
    
    if not results:
        return None
    
    df = pl.DataFrame(results)
    
    numeric_cols = ['syn_count', 'gaba_avg', 'ach_avg', 'glut_avg', 'oct_avg', 'ser_avg', 'da_avg']
    for col_name in numeric_cols:
        if col_name in df.columns:
            df = df.with_columns(pl.col(col_name).cast(pl.Float64, strict=False).fill_null(0))
    
    return df

def get_neuropil_statistics(_client):
    db = _client["connectome"]
    col = db["clean_connections"]
    
    pipeline = [
        {"$match": {
            "$and": [
                {"groups.neuropil": {"$ne": None}},
                {"$or": [
                    {"metrics.gaba_avg": {"$gt": 0}},
                    {"metrics.glut_avg": {"$gt": 0}}
                ]}
            ]
        }},
        {"$group": {
            "_id": "$groups.neuropil",
            "total_synapses": {"$sum": "$counts.syn_count"},
            "avg_gaba": {"$avg": "$metrics.gaba_avg"},
            "avg_glut": {"$avg": "$metrics.glut_avg"},
            "avg_ach": {"$avg": "$metrics.ach_avg"},
            "connection_count": {"$sum": 1}
        }},
        {"$match": {"total_synapses": {"$gt": 0}}},
        {"$sort": {"total_synapses": -1}},
        {"$limit": 50}
    ]
    
    results = list(col.aggregate(pipeline, allowDiskUse=True))
    
    if not results:
        return None
    
    df = pl.DataFrame(results)
    df = df.rename({"_id": "neuropil"})
    
    numeric_cols = ['total_synapses', 'avg_gaba', 'avg_glut', 'avg_ach', 'connection_count']
    for col_name in numeric_cols:
        if col_name in df.columns:
            df = df.with_columns(pl.col(col_name).cast(pl.Float64, strict=False).fill_null(0))
    
    return df

def build_connectivity_matrix(_client, sample_size=5000):
    db = _client["connectome"]
    col = db["clean_connections"]
    
    pipeline = [
        {"$match": {
            "$and": [
                {"groups.neuropil": {"$ne": None}},
                {"counts.syn_count": {"$gt": 0}}
            ]
        }},
        {"$limit": sample_size},
        {"$group": {
            "_id": "$groups.neuropil",
            "total_synapses": {"$sum": "$counts.syn_count"}
        }},
        {"$sort": {"total_synapses": -1}}
    ]
    
    results = list(col.aggregate(pipeline, allowDiskUse=True))
    
    if not results:
        return None
    
    processed = [{"neuropil": r["_id"], "total_synapses": r["total_synapses"]} for r in results]
    df = pl.DataFrame(processed)
    
    if 'total_synapses' in df.columns:
        df = df.with_columns(pl.col('total_synapses').cast(pl.Float64, strict=False))
    
    return df

def render_ingest_status():
    st.header("Raw Ingest Status — Connectome Data Analysis")
    
    client, mongo_ok = get_mongo_client()
    if not mongo_ok or client is None:
        st.error("MongoDB connection failed.")
        return
    
    try:
        db = client["connectome"]
        col = db["raw_connections"]
        
        total_docs = col.estimated_document_count()
        
        if total_docs == 0:
            st.warning("No documents in raw_connections. Run ingestion pipeline first.")
            return
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Connections", f"{total_docs:,}")
        
        with col2:
            stats = db.command(SON([("collStats", "raw_connections"), ("scale", 1024 * 1024)]))
            storage_mb = stats.get("storageSize", 0)
            st.metric("Storage Size", f"{storage_mb:.2f} MB")
        
        with col3:
            avg_obj_kb = stats.get("avgObjSize", 0) / 1024
            st.metric("Avg Connection Size", f"{avg_obj_kb:.2f} KB")
        
        with col4:
            compression_ratio = stats.get("storageSize", 1) / stats.get("size", 1) if stats.get("size", 0) > 0 else 1
            st.metric("Compression Ratio", f"{compression_ratio:.2f}x")
        
        st.divider()
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🧠 Connectivity Matrix",
            "🔬 Neurotransmitter Analysis",
            "📊 Neuropil Statistics",
            "🗺️ Regional Distribution",
            "🗂️ Browse Connections"
        ])
        
        with tab1:
            st.subheader("Neuropil Connectivity Matrix")
            
            matrix_data = build_connectivity_matrix(client, sample_size=5000)
            
            if matrix_data is not None and matrix_data.shape[0] > 0:
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    top_n = st.slider("Top N neuropils", 10, 50, 30)
                    
                    plot_data = matrix_data.head(top_n)
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        y=plot_data["neuropil"].to_list(),
                        x=plot_data["total_synapses"].to_list(),
                        orientation='h',
                        marker=dict(
                            color=plot_data["total_synapses"].to_list(),
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="Synapses")
                        ),
                        text=plot_data["total_synapses"].to_list(),
                        textposition='outside'
                    ))
                    
                    fig.update_layout(
                        title="Synaptic Density by Neuropil",
                        xaxis_title="Total Synapses (log scale)",
                        yaxis_title="Neuropil Region",
                        height=800,
                        xaxis_type="log",
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("#### Matrix Statistics")
                    
                    total_synapses = plot_data["total_synapses"].sum()
                    max_synapses = plot_data["total_synapses"].max()
                    min_synapses = plot_data["total_synapses"].min()
                    
                    st.metric("Total Synapses", f"{total_synapses:,}")
                    st.metric("Max (Region)", f"{max_synapses:,}")
                    st.metric("Min (Region)", f"{min_synapses:,}")
                    
                    st.divider()
                    
                    st.markdown("#### Top Regions")
                    for idx, row in enumerate(plot_data.head(5).iter_rows(named=True), 1):
                        st.write(f"{idx}. **{row['neuropil']}**: {row['total_synapses']:,}")
            else:
                st.info("Insufficient data for connectivity matrix.")
        
        with tab2:
            st.subheader("Neurotransmitter Distribution Analysis")
            
            nt_data = analyze_clean_connectome_data(client)
            
            if nt_data is not None and nt_data.shape[0] > 0:
                neurotransmitters = ['gaba_avg', 'glut_avg', 'ach_avg', 'da_avg', 'oct_avg', 'ser_avg']
                nt_names = ['GABA', 'Glutamate', 'Acetylcholine', 'Dopamine', 'Octopamine', 'Serotonin']
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    nt_means = []
                    for nt in neurotransmitters:
                        if nt in nt_data.columns:
                            mean_val = nt_data[nt].mean()
                            nt_means.append(mean_val if mean_val and not pl.Series([mean_val]).is_null()[0] else 0)
                        else:
                            nt_means.append(0)
                    
                    fig = go.Figure()
                    
                    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3']
                    
                    fig.add_trace(go.Bar(
                        x=nt_names,
                        y=nt_means,
                        marker=dict(
                            color=colors,
                            line=dict(color='#fff', width=2)
                        ),
                        text=[f"{v:.4f}" if v > 0 else "0.0000" for v in nt_means],
                        textposition='outside',
                        textfont=dict(size=12, color='white')
                    ))
                    
                    fig.update_layout(
                        title={
                            'text': "Average Neurotransmitter Concentrations",
                            'font': {'size': 18, 'color': 'white'}
                        },
                        xaxis_title="Neurotransmitter",
                        yaxis_title="Average Concentration",
                        height=450,
                        plot_bgcolor='#0e1117',
                        paper_bgcolor='#0e1117',
                        font=dict(color='white'),
                        xaxis=dict(gridcolor='#333', color='white'),
                        yaxis=dict(gridcolor='#333', color='white')
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("#### NT Statistics")
                    
                    for name, nt in zip(nt_names, neurotransmitters):
                        if nt in nt_data.columns:
                            avg_val = nt_data[nt].mean()
                            if avg_val and not pl.Series([avg_val]).is_null()[0]:
                                st.metric(name, f"{avg_val:.4f}")
                            else:
                                st.metric(name, "0.0000")
                        else:
                            st.metric(name, "N/A")
                
                st.divider()
                
                st.markdown("#### Neurotransmitter Distribution by Region")
                
                selected_nt = st.selectbox(
                    "Select neurotransmitter",
                    options=list(zip(nt_names, neurotransmitters)),
                    format_func=lambda x: x[0]
                )
                
                if selected_nt[1] in nt_data.columns:
                    valid_data = nt_data.filter(
                        (pl.col(selected_nt[1]).is_not_null()) & 
                        (pl.col(selected_nt[1]) > 0)
                    )
                    
                    if valid_data.shape[0] > 0:
                        fig = go.Figure()
                        
                        fig.add_trace(go.Histogram(
                            x=valid_data[selected_nt[1]].to_list(),
                            nbinsx=50,
                            marker=dict(
                                color='#EF553B',
                                line=dict(color='white', width=1)
                            )
                        ))
                        
                        fig.update_layout(
                            title=f"{selected_nt[0]} Concentration Distribution",
                            xaxis_title="Concentration",
                            yaxis_title="Frequency",
                            height=400,
                            plot_bgcolor='#0e1117',
                            paper_bgcolor='#0e1117',
                            font=dict(color='white'),
                            xaxis=dict(gridcolor='#333'),
                            yaxis=dict(gridcolor='#333')
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning(f"No valid data for {selected_nt[0]}")
                
                st.divider()
                
                st.markdown("#### Neurotransmitter Correlation Matrix")
                
                corr_data = nt_data.select(neurotransmitters)
                
                valid_corr = corr_data.filter(
                    pl.all_horizontal([pl.col(c).is_not_null() for c in neurotransmitters])
                )
                
                if valid_corr.shape[0] > 10:
                    corr_matrix = valid_corr.to_pandas().corr()
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=corr_matrix.values,
                        x=nt_names,
                        y=nt_names,
                        colorscale='RdBu',
                        zmid=0,
                        zmin=-1,
                        zmax=1,
                        text=corr_matrix.values,
                        texttemplate='%{text:.2f}',
                        textfont={"size": 12, "color": "white"},
                        colorbar=dict(
                            title=dict(text="Correlation", font=dict(color='white')),
                            tickfont=dict(color='white')
                        )
                    ))
                    
                    fig.update_layout(
                        title="Neurotransmitter Correlation Matrix",
                        height=550,
                        plot_bgcolor='#0e1117',
                        paper_bgcolor='#0e1117',
                        font=dict(color='white'),
                        xaxis=dict(color='white'),
                        yaxis=dict(color='white')
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Insufficient valid data for correlation analysis")
            else:
                st.info("Loading neurotransmitter data...")
        
        with tab3:
            st.subheader("Neuropil Region Statistics")
            
            neuropil_stats = get_neuropil_statistics(client)
            
            if neuropil_stats is not None:
                st.dataframe(
                    neuropil_stats.select([
                        pl.col("neuropil").alias("Region"),
                        pl.col("connection_count").alias("Connections"),
                        pl.col("total_synapses").alias("Total Synapses"),
                        pl.col("avg_gaba").alias("Avg GABA"),
                        pl.col("avg_glut").alias("Avg Glutamate"),
                        pl.col("avg_ach").alias("Avg ACh")
                    ]),
                    use_container_width=True,
                    height=400
                )
                
                st.divider()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.pie(
                        neuropil_stats.to_pandas(),
                        values='total_synapses',
                        names='neuropil',
                        title='Synaptic Distribution by Region (Top 50)',
                        hole=0.4
                    )
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=neuropil_stats["avg_gaba"].to_list(),
                        y=neuropil_stats["avg_glut"].to_list(),
                        mode='markers',
                        marker=dict(
                            size=neuropil_stats["total_synapses"].to_list(),
                            sizemode='area',
                            sizeref=2.*max(neuropil_stats["total_synapses"].to_list())/(40.**2),
                            color=neuropil_stats["connection_count"].to_list(),
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="Connections")
                        ),
                        text=neuropil_stats["neuropil"].to_list(),
                        hovertemplate='<b>%{text}</b><br>GABA: %{x:.4f}<br>Glutamate: %{y:.4f}<extra></extra>'
                    ))
                    
                    fig.update_layout(
                        title='GABA vs Glutamate by Region',
                        xaxis_title='Avg GABA',
                        yaxis_title='Avg Glutamate',
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Loading neuropil statistics...")
        
        with tab4:
            st.subheader("Regional Distribution Visualization")
            
            neuropil_stats = get_neuropil_statistics(client)
            
            if neuropil_stats is not None and neuropil_stats.shape[0] > 0:
                top_regions = neuropil_stats.head(20)
                
                st.markdown("#### 3D Neurotransmitter Space")
                st.caption("Top 20 regions plotted in GABA-Glutamate-Acetylcholine space")
                
                gaba_vals = top_regions["avg_gaba"].to_list()
                glut_vals = top_regions["avg_glut"].to_list()
                ach_vals = top_regions["avg_ach"].to_list()
                synapse_vals = top_regions["total_synapses"].to_list()
                conn_vals = top_regions["connection_count"].to_list()
                region_names = top_regions["neuropil"].to_list()
                
                max_synapses = max(synapse_vals) if synapse_vals else 1
                marker_sizes = [max(5, min(30, (s / max_synapses) * 30)) for s in synapse_vals]
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter3d(
                    x=gaba_vals,
                    y=glut_vals,
                    z=ach_vals,
                    mode='markers+text',
                    marker=dict(
                        size=marker_sizes,
                        color=conn_vals,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(
                            title=dict(text="Connections", font=dict(color='white')),
                            tickfont=dict(color='white')
                        ),
                        line=dict(color='white', width=0.5)
                    ),
                    text=region_names,
                    textposition="top center",
                    textfont=dict(size=9, color='white'),
                    hovertemplate='<b>%{text}</b><br>' +
                                  'GABA: %{x:.4f}<br>' +
                                  'Glutamate: %{y:.4f}<br>' +
                                  'ACh: %{z:.4f}<br>' +
                                  'Synapses: %{marker.size}<br>' +
                                  '<extra></extra>'
                ))
                
                fig.update_layout(
                    title={
                        'text': '3D Neurotransmitter Space (Top 20 Regions)',
                        'font': {'size': 18, 'color': 'white'}
                    },
                    scene=dict(
                        xaxis=dict(
                            title=dict(text='GABA', font=dict(color='white')),
                            backgroundcolor='#0e1117',
                            gridcolor='#333',
                            showbackground=True,
                            tickfont=dict(color='white')
                        ),
                        yaxis=dict(
                            title=dict(text='Glutamate', font=dict(color='white')),
                            backgroundcolor='#0e1117',
                            gridcolor='#333',
                            showbackground=True,
                            tickfont=dict(color='white')
                        ),
                        zaxis=dict(
                            title=dict(text='Acetylcholine', font=dict(color='white')),
                            backgroundcolor='#0e1117',
                            gridcolor='#333',
                            showbackground=True,
                            tickfont=dict(color='white')
                        ),
                        bgcolor='#0e1117'
                    ),
                    height=700,
                    paper_bgcolor='#0e1117',
                    font=dict(color='white')
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.divider()
                
                st.markdown("#### Region Statistics Table")
                
                display_df = top_regions.select([
                    pl.col("neuropil").alias("Region"),
                    pl.col("connection_count").cast(pl.Int64).alias("Connections"),
                    pl.col("total_synapses").cast(pl.Int64).alias("Synapses"),
                    pl.col("avg_gaba").alias("GABA"),
                    pl.col("avg_glut").alias("Glutamate"),
                    pl.col("avg_ach").alias("ACh")
                ])
                
                st.dataframe(display_df, use_container_width=True, height=400)
            else:
                st.info("Loading regional distribution data...")
        
        with tab5:
            st.subheader("Browse Raw Connections")
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                page_size = st.selectbox(
                    "Documents per page",
                    [10, 25, 50, 100],
                    index=1
                )
            
            with col2:
                max_pages = (total_docs // page_size) + 1
                current_page = st.number_input(
                    "Page",
                    min_value=1,
                    max_value=max_pages,
                    value=1,
                    step=1
                )
            
            with col3:
                st.metric("Total Pages", f"{max_pages:,}")
            
            skip = (current_page - 1) * page_size
            
            docs = get_paginated_raw_data(client, skip=skip, limit=page_size)
            
            if docs:
                st.markdown(f"#### Showing connections {skip + 1} to {skip + len(docs)} of {total_docs:,}")
                
                for idx, doc in enumerate(docs, start=skip + 1):
                    with st.expander(f"Connection #{idx}"):
                        
                        if isinstance(doc, dict) and 'data' in doc:
                            data = doc['data']
                            
                            col1, col2 = st.columns([1, 1])
                            
                            with col1:
                                st.markdown("##### Connection Details")
                                st.write(f"**Neuropil**: {data.get('neuropil', 'N/A')}")
                                st.write(f"**Synapse Count**: {data.get('syn_count', 0)}")
                                st.write(f"**Pre-synaptic ID**: {data.get('pre_pt_root_id', 'N/A')}")
                                st.write(f"**Post-synaptic ID**: {data.get('post_pt_root_id', 'N/A')}")
                                
                                if 'meta' in doc:
                                    st.markdown("##### Metadata")
                                    st.write(f"**Source**: {doc['meta'].get('source_file', 'N/A')}")
                                    st.write(f"**Ingested**: {doc['meta'].get('ingest_time', 'N/A')}")
                                
                                st.markdown("---")
                                if st.button(f"Show JSON #{idx}", key=f"json_btn_{idx}"):
                                    st.json(doc, expanded=True)
                            
                            with col2:
                                st.markdown("##### Neurotransmitters")
                                
                                nt_data = {
                                    'GABA': data.get('gaba_avg', 0),
                                    'Glutamate': data.get('glut_avg', 0),
                                    'Acetylcholine': data.get('ach_avg', 0),
                                    'Dopamine': data.get('da_avg', 0),
                                    'Octopamine': data.get('oct_avg', 0),
                                    'Serotonin': data.get('ser_avg', 0)
                                }
                                
                                fig = go.Figure(data=[
                                    go.Bar(
                                        x=list(nt_data.keys()),
                                        y=list(nt_data.values()),
                                        marker_color=['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3']
                                    )
                                ])
                                
                                fig.update_layout(
                                    title=f"NT Profile - {data.get('neuropil', 'Unknown')}",
                                    height=300,
                                    showlegend=False
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col2:
                    progress = (current_page / max_pages) * 100
                    st.progress(progress / 100, text=f"Page {current_page} of {max_pages}")
            else:
                st.info("No documents found for this page.")
        
    except Exception as e:
        st.error(f"Error accessing raw layer: {e}")
        st.exception(e)

def render_clean_layer():
    st.header("Clean Layer Analysis — Schema-aware Processing")
    
    client, mongo_ok = get_mongo_client()
    if not mongo_ok or client is None:
        st.error("MongoDB connection failed.")
        return
    
    try:
        db = client["connectome"]
        raw = db["raw_connections"]
        clean = db["clean_connections"]
        
        raw_n = raw.estimated_document_count()
        clean_n = clean.estimated_document_count()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Raw Documents", f"{raw_n:,}")
        
        with col2:
            st.metric("Clean Documents", f"{clean_n:,}")
        
        with col3:
            retention = (clean_n / raw_n * 100) if raw_n > 0 else 0
            st.metric("Retention Rate", f"{retention:.2f}%")
        
        with col4:
            dropped = raw_n - clean_n
            st.metric("Dropped Records", f"{dropped:,}")
        
        st.divider()
        
        if clean_n == 0:
            st.warning("clean_connections is empty. Run: `python -m app.clean.clean_connections`")
            return
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Schema Structure",
            "🔬 Entity Analysis",
            "📈 Metrics Distribution",
            "🗂️ Browse Clean Data"
        ])
        
        with tab1:
            st.subheader("Clean Layer Schema Structure")
            
            sample = clean.find_one()
            if sample:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("#### Schema Breakdown")
                    
                    schema_info = {
                        "entities": {
                            "description": "Source/target neuron identifiers",
                            "fields": list(sample.get('entities', {}).keys()),
                            "count": len(sample.get('entities', {}))
                        },
                        "groups": {
                            "description": "Categorical groupings (neuropil regions)",
                            "fields": list(sample.get('groups', {}).keys()),
                            "count": len(sample.get('groups', {}))
                        },
                        "counts": {
                            "description": "Integer count fields (synapse counts)",
                            "fields": list(sample.get('counts', {}).keys()),
                            "count": len(sample.get('counts', {}))
                        },
                        "metrics": {
                            "description": "Float metric fields (neurotransmitter concentrations)",
                            "fields": list(sample.get('metrics', {}).keys()),
                            "count": len(sample.get('metrics', {}))
                        },
                        "stats": {
                            "description": "Derived statistics (entropy, max, count)",
                            "fields": list(sample.get('stats', {}).keys()),
                            "count": len(sample.get('stats', {}))
                        }
                    }
                    
                    for category, info in schema_info.items():
                        with st.container():
                            st.markdown(f"**{category.upper()}** ({info['count']} fields)")
                            st.caption(info['description'])
                            if info['fields']:
                                st.code(", ".join(info['fields'][:10]))
                            st.divider()
                
                with col2:
                    st.markdown("#### Schema Statistics")
                    
                    total_fields = sum(info['count'] for info in schema_info.values())
                    st.metric("Total Field Categories", 5)
                    st.metric("Total Fields", total_fields)
                    
                    st.divider()
                    
                    st.markdown("#### Category Distribution")
                    
                    fig = go.Figure(data=[
                        go.Pie(
                            labels=list(schema_info.keys()),
                            values=[info['count'] for info in schema_info.values()],
                            hole=0.4
                        )
                    ])
                    
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                st.divider()
                
                st.markdown("#### Sample Clean Document")
                st.json(sample, expanded=False)
        
        with tab2:
            st.subheader("Entity Coverage Analysis")
            
            pipeline = [
                {"$project": {
                    "has_source": {"$cond": [{"$ne": ["$entities.source", None]}, 1, 0]},
                    "has_target": {"$cond": [{"$ne": ["$entities.target", None]}, 1, 0]},
                }},
                {"$group": {
                    "_id": None,
                    "source_present": {"$sum": "$has_source"},
                    "target_present": {"$sum": "$has_target"},
                    "n": {"$sum": 1},
                }}
            ]
            
            result = list(clean.aggregate(pipeline, allowDiskUse=True))
            
            if result:
                data = result[0]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    source_pct = (data["source_present"] / data["n"] * 100) if data["n"] > 0 else 0
                    st.metric(
                        "Source Entities",
                        f'{data["source_present"]:,}',
                        delta=f"{source_pct:.2f}%"
                    )
                
                with col2:
                    target_pct = (data["target_present"] / data["n"] * 100) if data["n"] > 0 else 0
                    st.metric(
                        "Target Entities",
                        f'{data["target_present"]:,}',
                        delta=f"{target_pct:.2f}%"
                    )
                
                with col3:
                    both_pct = min(source_pct, target_pct)
                    st.metric(
                        "Complete Pairs",
                        f"{both_pct:.2f}%"
                    )
                
                st.divider()
                
                st.markdown("#### Entity Distribution")
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=['Source', 'Target'],
                        y=[source_pct, target_pct],
                        marker_color=['#636EFA', '#EF553B'],
                        text=[f"{source_pct:.1f}%", f"{target_pct:.1f}%"],
                        textposition='outside'
                    )
                ])
                
                fig.update_layout(
                    title="Entity Presence Rate",
                    yaxis_title="Coverage %",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Metrics Distribution Analysis")
            
            pipeline = [
                {"$sample": {"size": 5000}},
                {"$project": {
                    "metric_count": "$stats.metric_count",
                    "metric_max": "$stats.metric_max",
                    "metric_entropy": "$stats.metric_entropy"
                }}
            ]
            
            results = list(clean.aggregate(pipeline, allowDiskUse=True))
            
            if results:
                df = pl.DataFrame(results)
                
                if 'metric_count' in df.columns:
                    df = df.with_columns(pl.col('metric_count').cast(pl.Float64, strict=False))
                if 'metric_max' in df.columns:
                    df = df.with_columns(pl.col('metric_max').cast(pl.Float64, strict=False))
                if 'metric_entropy' in df.columns:
                    df = df.with_columns(pl.col('metric_entropy').cast(pl.Float64, strict=False))
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    avg_count = df["metric_count"].mean() if "metric_count" in df.columns else 0
                    st.metric("Avg Metrics/Doc", f"{avg_count:.2f}")
                
                with col2:
                    avg_max = df["metric_max"].mean() if "metric_max" in df.columns else 0
                    st.metric("Avg Max Value", f"{avg_max:.4f}")
                
                with col3:
                    avg_entropy = df["metric_entropy"].mean() if "metric_entropy" in df.columns else 0
                    st.metric("Avg Entropy", f"{avg_entropy:.4f}")
                
                st.divider()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if "metric_count" in df.columns:
                        fig = go.Figure()
                        fig.add_trace(go.Histogram(
                            x=df["metric_count"].to_list(),
                            nbinsx=30,
                            marker_color='#636EFA'
                        ))
                        
                        fig.update_layout(
                            title="Metrics Count Distribution",
                            xaxis_title="Number of Metrics",
                            yaxis_title="Frequency",
                            height=350
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    if "metric_entropy" in df.columns:
                        fig = go.Figure()
                        fig.add_trace(go.Histogram(
                            x=df["metric_entropy"].to_list(),
                            nbinsx=30,
                            marker_color='#EF553B'
                        ))
                        
                        fig.update_layout(
                            title="Metric Entropy Distribution",
                            xaxis_title="Entropy",
                            yaxis_title="Frequency",
                            height=350
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                st.divider()
                
                st.markdown("#### Metrics Statistics Table")
                
                stats_df = df.describe()
                st.dataframe(stats_df, use_container_width=True)
        
        with tab4:
            st.subheader("Browse Clean Data")
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                page_size = st.selectbox(
                    "Documents per page",
                    [10, 25, 50, 100],
                    index=1
                )
            
            with col2:
                max_pages = (clean_n // page_size) + 1
                current_page = st.number_input(
                    "Page",
                    min_value=1,
                    max_value=max_pages,
                    value=1,
                    step=1
                )
            
            with col3:
                st.metric("Total Pages", f"{max_pages:,}")
            
            skip = (current_page - 1) * page_size
            
            docs = list(clean.find({}).skip(skip).limit(page_size))
            
            if docs:
                st.markdown(f"#### Showing documents {skip + 1} to {skip + len(docs)} of {clean_n:,}")
                
                for idx, doc in enumerate(docs, start=skip + 1):
                    with st.expander(f"Clean Document #{idx}"):
                        
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.markdown("##### Entities")
                            entities = doc.get('entities', {})
                            for k, v in entities.items():
                                st.write(f"**{k}**: {v}")
                            
                            st.divider()
                            
                            st.markdown("##### Groups")
                            groups = doc.get('groups', {})
                            for k, v in groups.items():
                                st.write(f"**{k}**: {v}")
                            
                            st.divider()
                            
                            st.markdown("##### Counts")
                            counts = doc.get('counts', {})
                            for k, v in counts.items():
                                st.write(f"**{k}**: {v}")
                        
                        with col2:
                            st.markdown("##### Metrics")
                            metrics = doc.get('metrics', {})
                            
                            if metrics:
                                fig = go.Figure(data=[
                                    go.Bar(
                                        x=list(metrics.keys()),
                                        y=list(metrics.values()),
                                        marker_color='#00CC96'
                                    )
                                ])
                                
                                fig.update_layout(
                                    title="Metric Values",
                                    height=250,
                                    showlegend=False
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            
                            st.divider()
                            
                            st.markdown("##### Statistics")
                            stats = doc.get('stats', {})
                            for k, v in stats.items():
                                if isinstance(v, float):
                                    st.write(f"**{k}**: {v:.4f}")
                                else:
                                    st.write(f"**{k}**: {v}")
                        
                        if st.button(f"Show Full JSON #{idx}", key=f"clean_json_{idx}"):
                            st.json(doc, expanded=True)
                
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col2:
                    progress = (current_page / max_pages) * 100
                    st.progress(progress / 100, text=f"Page {current_page} of {max_pages}")
            else:
                st.info("No documents found for this page.")
        
    except Exception as e:
        st.error(f"Error accessing clean layer: {e}")
        st.exception(e)
def render_gold_analytics():
    st.header("Gold Analytics")
    
    client, ok = get_mongo_client()
    if not ok or not client:
        st.error("MongoDB offline")
        return
    
    db = client["connectome"]
    
    try:
        region_n = db["gold_region_agg"].estimated_document_count()
        tensor_n = db["gold_tensor"].estimated_document_count()
    except:
        region_n = tensor_n = 0
    
    if region_n == 0 and tensor_n == 0:
        st.warning("Gold layer empty")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Region Agg", f"{region_n:,}")
    
    with col2:
        st.metric("Tensor Docs", f"{tensor_n:,}")
    
    stats_region = db.command(SON([("collStats", "gold_region_agg"), ("scale", 1024)]))
    stats_tensor = db.command(SON([("collStats", "gold_tensor"), ("scale", 1024)]))
    
    with col3:
        total_kb = stats_region.get("storageSize", 0) + stats_tensor.get("storageSize", 0)
        st.metric("Storage", f"{total_kb:.1f} KB")
    
    with col4:
        compression = (stats_region.get("storageSize", 1) / stats_region.get("size", 1)) if stats_region.get("size", 0) > 0 else 0
        st.metric("Compression", f"{compression:.2f}x")
    
    st.divider()
    
    tab1, tab2, tab3 = st.tabs(["Region Agg", "Tensor Space", "Analytics"])
    
    with tab1:
        pipeline = [
            {"$match": {"total_syn": {"$gt": 0}}},
            {"$sort": {"total_syn": -1}},
            {"$limit": 30},
            {"$project": {
                "source": "$source_region",
                "target": "$target_region",
                "edges": "$edge_count",
                "synapses": "$total_syn",
                "entropy": "$avg_entropy"
            }}
        ]
        
        results = list(db["gold_region_agg"].aggregate(pipeline, allowDiskUse=True))
        
        if results:
            sources = [r.get("source", "?") for r in results]
            targets = [r.get("target", "?") for r in results]
            synapses = [r.get("synapses", 0) for r in results]
            edges = [r.get("edges", 0) for r in results]
            entropy = [r.get("entropy", 0) for r in results]
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=[f"{s}→{t}" for s, t in zip(sources[:15], targets[:15])],
                    y=synapses[:15],
                    marker=dict(
                        color=synapses[:15],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(
                            title=dict(text="Synapses", font=dict(color='white')),
                            tickfont=dict(color='white')
                        )
                    ),
                    text=[f"{s:,}" for s in synapses[:15]],
                    textposition='outside'
                ))
                
                fig.update_layout(
                    title="Top Region Pairs",
                    height=400,
                    plot_bgcolor='#0e1117',
                    paper_bgcolor='#0e1117',
                    font=dict(color='white'),
                    xaxis=dict(tickangle=-45, color='white'),
                    yaxis=dict(gridcolor='#333', color='white', type='log')
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig2 = go.Figure()
                
                fig2.add_trace(go.Scatter(
                    x=edges,
                    y=synapses,
                    mode='markers',
                    marker=dict(
                        size=15,
                        color=entropy,
                        colorscale='Plasma',
                        showscale=True,
                        colorbar=dict(
                            title=dict(text="Entropy", font=dict(color='white')),
                            tickfont=dict(color='white')
                        ),
                        line=dict(color='white', width=1)
                    ),
                    text=[f"{s}→{t}" for s, t in zip(sources, targets)],
                    hovertemplate='<b>%{text}</b><br>Edges: %{x}<br>Synapses: %{y}<br>Entropy: %{marker.color:.3f}<extra></extra>'
                ))
                
                fig2.update_layout(
                    title="Edges vs Synapses",
                    xaxis_title="Edge Count",
                    yaxis_title="Synapse Count",
                    height=400,
                    plot_bgcolor='#0e1117',
                    paper_bgcolor='#0e1117',
                    font=dict(color='white'),
                    xaxis=dict(gridcolor='#333', color='white', type='log'),
                    yaxis=dict(gridcolor='#333', color='white', type='log')
                )
                
                st.plotly_chart(fig2, use_container_width=True)
            
            st.divider()
            
            df = pl.DataFrame(results[:20])
            st.dataframe(df, use_container_width=True, hide_index=True, height=350)
    
    with tab2:
        pipeline_tensor = [
            {"$match": {"value": {"$ne": None, "$exists": True}}},
            {"$limit": 3000},
            {"$project": {
                "src": "$coords.src",
                "tgt": "$coords.tgt",
                "metric": "$coords.metric",
                "value": 1,
                "support": 1
            }},
            {"$match": {"src": {"$ne": None}, "value": {"$gt": 0}}}
        ]
        
        results = list(db["gold_tensor"].aggregate(pipeline_tensor, allowDiskUse=True))
        
        if results:
            metrics_dict = {}
            for r in results:
                metric = r.get("metric", "unknown")
                val = r.get("value", 0)
                if isinstance(val, (int, float)) and val > 0:
                    if metric not in metrics_dict:
                        metrics_dict[metric] = []
                    metrics_dict[metric].append(val)
            
            if metrics_dict:
                col1, col2 = st.columns(2)
                
                with col1:
                    metric_names = list(metrics_dict.keys())
                    metric_counts = [len(v) for v in metrics_dict.values()]
                    
                    fig = go.Figure(go.Bar(
                        x=metric_names,
                        y=metric_counts,
                        marker=dict(color=metric_counts, colorscale='Blues'),
                        text=[f"{c:,}" for c in metric_counts],
                        textposition='outside'
                    ))
                    
                    fig.update_layout(
                        title="Metric Dimension Coverage",
                        xaxis_title="Metric",
                        yaxis_title="Entry Count",
                        height=400,
                        plot_bgcolor='#0e1117',
                        paper_bgcolor='#0e1117',
                        font=dict(color='white'),
                        xaxis=dict(tickangle=-45, color='white'),
                        yaxis=dict(gridcolor='#333', color='white')
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    selected_metric = st.selectbox("Select Metric", metric_names)
                    
                    if selected_metric in metrics_dict:
                        values = metrics_dict[selected_metric]
                        
                        fig2 = go.Figure()
                        
                        fig2.add_trace(go.Histogram(
                            x=values,
                            nbinsx=50,
                            marker=dict(
                                color='#ef553b',
                                line=dict(color='white', width=1)
                            )
                        ))
                        
                        fig2.update_layout(
                            title=f"{selected_metric} Distribution",
                            xaxis_title="Value",
                            yaxis_title="Frequency",
                            height=400,
                            plot_bgcolor='#0e1117',
                            paper_bgcolor='#0e1117',
                            font=dict(color='white'),
                            xaxis=dict(gridcolor='#333', color='white'),
                            yaxis=dict(gridcolor='#333', color='white')
                        )
                        
                        st.plotly_chart(fig2, use_container_width=True)
            
            st.divider()
            
            src_counts = {}
            for r in results:
                src = r.get("src")
                if src:
                    src_counts[src] = src_counts.get(src, 0) + 1
            
            top_src = sorted(src_counts.items(), key=lambda x: x[1], reverse=True)[:15]
            
            if top_src:
                fig3 = go.Figure()
                
                fig3.add_trace(go.Bar(
                    x=[s[0] for s in top_src],
                    y=[s[1] for s in top_src],
                    marker=dict(
                        color=[s[1] for s in top_src],
                        colorscale='Greens'
                    ),
                    text=[f"{s[1]:,}" for s in top_src],
                    textposition='outside'
                ))
                
                fig3.update_layout(
                    title="Top Source Regions (Tensor Space)",
                    height=350,
                    plot_bgcolor='#0e1117',
                    paper_bgcolor='#0e1117',
                    font=dict(color='white'),
                    xaxis=dict(tickangle=-45, color='white'),
                    yaxis=dict(gridcolor='#333', color='white')
                )
                
                st.plotly_chart(fig3, use_container_width=True)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Region Aggregation")
            
            agg_pipeline = [
                {"$group": {
                    "_id": None,
                    "total_edges": {"$sum": "$edge_count"},
                    "total_syn": {"$sum": "$total_syn"},
                    "avg_entropy": {"$avg": "$avg_entropy"},
                    "regions": {"$sum": 1}
                }}
            ]
            
            agg_stats = list(db["gold_region_agg"].aggregate(agg_pipeline, allowDiskUse=True))
            
            if agg_stats and agg_stats[0]:
                stats = agg_stats[0]
                
                metrics = [
                    {"Metric": "Region Pairs", "Value": f"{stats.get('regions', 0):,}"},
                    {"Metric": "Total Edges", "Value": f"{stats.get('total_edges', 0):,}"},
                    {"Metric": "Total Synapses", "Value": f"{stats.get('total_syn', 0):,}"},
                    {"Metric": "Avg Entropy", "Value": f"{stats.get('avg_entropy', 0):.4f}"}
                ]
                
                df = pl.DataFrame(metrics)
                st.dataframe(df, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("#### Tensor Metrics")
            
            tensor_pipeline = [
                {"$match": {"value": {"$ne": None, "$exists": True}}},
                {"$group": {
                    "_id": None,
                    "total_entries": {"$sum": 1},
                    "avg_value": {"$avg": "$value"},
                    "max_value": {"$max": "$value"},
                    "min_value": {"$min": "$value"},
                    "total_support": {"$sum": "$support"}
                }}
            ]
            
            tensor_stats = list(db["gold_tensor"].aggregate(tensor_pipeline, allowDiskUse=True))
            
            if tensor_stats and tensor_stats[0]:
                stats = tensor_stats[0]
                
                metrics = [
                    {"Metric": "Tensor Entries", "Value": f"{stats.get('total_entries', 0):,}"},
                    {"Metric": "Total Support", "Value": f"{stats.get('total_support', 0):,}"},
                    {"Metric": "Avg Value", "Value": f"{stats.get('avg_value', 0):.6f}"},
                    {"Metric": "Value Range", "Value": f"[{stats.get('min_value', 0):.6f}, {stats.get('max_value', 0):.6f}]"}
                ]
                
                df = pl.DataFrame(metrics)
                st.dataframe(df, use_container_width=True, hide_index=True)
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            entropy_pipeline = [
                {"$match": {"avg_entropy": {"$ne": None}}},
                {"$limit": 100},
                {"$project": {"entropy": "$avg_entropy"}}
            ]
            
            entropy_results = list(db["gold_region_agg"].aggregate(entropy_pipeline, allowDiskUse=True))
            
            if entropy_results:
                entropy_vals = [r.get("entropy", 0) for r in entropy_results if isinstance(r.get("entropy"), (int, float))]
                
                if entropy_vals:
                    fig = go.Figure()
                    
                    fig.add_trace(go.Histogram(
                        x=entropy_vals,
                        nbinsx=30,
                        marker=dict(color='#636efa', line=dict(color='white', width=1))
                    ))
                    
                    fig.update_layout(
                        title="Entropy Distribution",
                        xaxis_title="Entropy",
                        yaxis_title="Frequency",
                        height=300,
                        plot_bgcolor='#0e1117',
                        paper_bgcolor='#0e1117',
                        font=dict(color='white'),
                        xaxis=dict(gridcolor='#333', color='white'),
                        yaxis=dict(gridcolor='#333', color='white')
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            support_pipeline = [
                {"$match": {"support": {"$ne": None, "$gt": 0}}},
                {"$limit": 100},
                {"$project": {"support": 1}}
            ]
            
            support_results = list(db["gold_tensor"].aggregate(support_pipeline, allowDiskUse=True))
            
            if support_results:
                support_vals = [r.get("support", 0) for r in support_results if isinstance(r.get("support"), (int, float))]
                
                if support_vals:
                    fig = go.Figure()
                    
                    fig.add_trace(go.Box(
                        y=support_vals,
                        marker=dict(color='#00cc96'),
                        name="Support"
                    ))
                    
                    fig.update_layout(
                        title="Support Distribution",
                        yaxis_title="Support",
                        height=300,
                        plot_bgcolor='#0e1117',
                        paper_bgcolor='#0e1117',
                        font=dict(color='white'),
                        yaxis=dict(gridcolor='#333', color='white')
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    st.success("Gold layer validated")

@st.cache_data(ttl=30)
def get_pipeline_metrics(_client):
    db = _client["connectome"]
    
    collections = {
        "raw_connections": "Raw",
        "clean_connections": "Clean",
        "gold_region_agg": "Gold:Region",
        "gold_tensor": "Gold:Tensor"
    }
    
    metrics = []
    
    for col_name, label in collections.items():
        try:
            col = db[col_name]
            count = col.estimated_document_count()
            
            stats = db.command(SON([
                ("collStats", col_name),
                ("scale", 1024 * 1024)
            ]))
            
            metrics.append({
                "collection": col_name,
                "label": label,
                "documents": count,
                "storage_mb": round(stats.get("storageSize", 0), 2),
                "avg_doc_kb": round(stats.get("avgObjSize", 0) / 1024, 2) if stats.get("avgObjSize") else 0,
                "status": "ready" if count > 0 else "empty"
            })
        except Exception as e:
            metrics.append({
                "collection": col_name,
                "label": label,
                "documents": 0,
                "storage_mb": 0,
                "avg_doc_kb": 0,
                "status": "error"
            })
    
    return pl.DataFrame(metrics)

@st.cache_data(ttl=60)
def analyze_pipeline_efficiency(_client):
    db = _client["connectome"]
    
    raw_col = db["raw_connections"]
    clean_col = db["clean_connections"]
    
    raw_n = raw_col.estimated_document_count()
    clean_n = clean_col.estimated_document_count()
    
    if raw_n == 0:
        return None
    
    pipeline = [
        {"$sample": {"size": 1000}},
        {"$project": {
            "field_count": {"$size": {"$objectToArray": "$$ROOT"}},
            "has_weight": {"$cond": [{"$ifNull": ["$weight", False]}, 1, 0]}
        }},
        {"$group": {
            "_id": None,
            "avg_fields": {"$avg": "$field_count"},
            "weight_coverage": {"$avg": "$has_weight"}
        }}
    ]
    
    raw_stats = list(raw_col.aggregate(pipeline, allowDiskUse=True))
    clean_stats = list(clean_col.aggregate(pipeline, allowDiskUse=True))
    
    return {
        "retention_rate": (clean_n / raw_n) * 100,
        "dropped_records": raw_n - clean_n,
        "raw_avg_fields": raw_stats[0]["avg_fields"] if raw_stats else 0,
        "clean_avg_fields": clean_stats[0]["avg_fields"] if clean_stats else 0,
        "field_enrichment": (clean_stats[0]["avg_fields"] - raw_stats[0]["avg_fields"]) if (raw_stats and clean_stats) else 0
    }

@st.cache_data(ttl=120)
def get_neo4j_status(_driver):
    try:
        with _driver.session() as session:
            node_result = session.run("MATCH (n) RETURN count(n) as count")
            node_count = node_result.single()["count"]
            
            rel_result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            rel_count = rel_result.single()["count"]
            
            return {
                "nodes": node_count,
                "relationships": rel_count,
                "status": "ready" if node_count > 0 else "empty"
            }
    except Exception:
        return {"nodes": 0, "relationships": 0, "status": "offline"}
def render_pipelines():
    st.header("Pipeline Control")
    
    client, mongo_ok = get_mongo_client()
    driver, neo4j_ok = get_neo4j_driver()
    
    if not mongo_ok or not client:
        st.error("MongoDB offline")
        return
    
    db = client["connectome"]
    
    raw_n = db["raw_connections"].estimated_document_count()
    clean_n = db["clean_connections"].estimated_document_count()
    
    try:
        region_n = db["gold_region_agg"].estimated_document_count()
        tensor_n = db["gold_tensor"].estimated_document_count()
    except:
        region_n = tensor_n = 0
    
    neo4j_nodes = 0
    neo4j_rels = 0
    
    if neo4j_ok and driver:
        try:
            with driver.session() as session:
                neo4j_nodes = session.run("MATCH (n) RETURN count(n) as c").single()["c"]
                neo4j_rels = session.run("MATCH ()-[r]->() RETURN count(r) as c").single()["c"]
        except:
            pass
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Raw", f"{raw_n/1e6:.2f}M")
    with col2:
        retention = (clean_n/raw_n*100) if raw_n > 0 else 0
        st.metric("Clean", f"{clean_n/1e6:.2f}M", f"{retention:.1f}%")
    with col3:
        st.metric("Region", f"{region_n:,}")
    with col4:
        st.metric("Tensor", f"{tensor_n:,}")
    with col5:
        st.metric("Graph", f"{neo4j_nodes:,}n/{neo4j_rels:,}e")
    
    st.divider()
    
    tab1, tab2, tab3 = st.tabs(["Topology", "Metrics", "Flow"])
    
    with tab1:
        nodes = [
            {"id": "raw", "label": f"Raw\n{raw_n/1e6:.1f}M", "r": 40, "status": "ready" if raw_n > 0 else "empty"},
            {"id": "clean", "label": f"Clean\n{clean_n/1e6:.1f}M", "r": 40, "status": "ready" if clean_n > 0 else "empty"},
            {"id": "region", "label": f"Region\n{region_n}", "r": 35, "status": "ready" if region_n > 0 else "empty"},
            {"id": "tensor", "label": f"Tensor\n{tensor_n}", "r": 35, "status": "ready" if tensor_n > 0 else "empty"},
            {"id": "neo4j", "label": f"Neo4j\n{neo4j_nodes}", "r": 40, "status": "ready" if neo4j_nodes > 0 else "empty"}
        ]
        
        links = [
            {"source": "raw", "target": "clean"},
            {"source": "clean", "target": "region"},
            {"source": "clean", "target": "tensor"},
            {"source": "tensor", "target": "neo4j"}
        ]
        
        html = f"""
        <svg width="100%" height="500" style="background:#0e1117">
          <defs>
            <marker id="arrow" viewBox="0 -5 10 10" refX="35" markerWidth="6" markerHeight="6" orient="auto">
              <path d="M0,-5L10,0L0,5" fill="#666"/>
            </marker>
            <filter id="glow">
              <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
              <feMerge>
                <feMergeNode in="coloredBlur"/>
                <feMergeNode in="SourceGraphic"/>
              </feMerge>
            </filter>
          </defs>
          <script src="https://d3js.org/d3.v7.min.js"></script>
          <script>
            const data = {{nodes: {json.dumps(nodes)}, links: {json.dumps(links)}}};
            const svg = d3.select("svg");
            const width = svg.node().getBoundingClientRect().width;
            const height = 500;
            
            const colorMap = {{ready: "#00cc96", empty: "#636efa", error: "#ef553b"}};
            
            const simulation = d3.forceSimulation(data.nodes)
              .force("link", d3.forceLink(data.links).id(d => d.id).distance(180))
              .force("charge", d3.forceManyBody().strength(-800))
              .force("center", d3.forceCenter(width/2, height/2))
              .force("collision", d3.forceCollide().radius(50));
            
            const link = svg.append("g").selectAll("line")
              .data(data.links).enter().append("line")
              .attr("stroke", "#666").attr("stroke-width", 3)
              .attr("marker-end", "url(#arrow)");
            
            const node = svg.append("g").selectAll("g")
              .data(data.nodes).enter().append("g")
              .call(d3.drag()
                .on("start", (e,d) => {{ if(!e.active) simulation.alphaTarget(0.3).restart(); d.fx=d.x; d.fy=d.y; }})
                .on("drag", (e,d) => {{ d.fx=e.x; d.fy=e.y; }})
                .on("end", (e,d) => {{ if(!e.active) simulation.alphaTarget(0); d.fx=d.fy=null; }})
              );
            
            node.append("circle")
              .attr("r", d => d.r)
              .attr("fill", d => colorMap[d.status])
              .attr("stroke", "#fff").attr("stroke-width", 3)
              .attr("filter", "url(#glow)");
            
            node.append("text").attr("text-anchor", "middle").attr("fill", "#fff")
              .attr("font-size", "13px").attr("font-weight", "600")
              .selectAll("tspan").data(d => d.label.split("\\n")).enter()
              .append("tspan").attr("x", 0).attr("dy", (d,i) => i*16-8).text(d => d);
            
            simulation.on("tick", () => {{
              link.attr("x1", d => d.source.x).attr("y1", d => d.source.y)
                  .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
              node.attr("transform", d => `translate(${{d.x}},${{d.y}})`);
            }});
          </script>
        </svg>
        """
        
        components.html(html, height=520)
    
    with tab2:
        pipeline = [
            {"$match": {"groups.neuropil": {"$ne": None}, "counts.syn_count": {"$gt": 0}}},
            {"$limit": 5000},
            {"$group": {
                "_id": "$groups.neuropil",
                "connections": {"$sum": 1},
                "synapses": {"$sum": "$counts.syn_count"},
                "gaba": {"$avg": "$metrics.gaba_avg"},
                "glut": {"$avg": "$metrics.glut_avg"}
            }},
            {"$match": {"synapses": {"$gt": 0}}},
            {"$sort": {"synapses": -1}},
            {"$limit": 30}
        ]
        
        results = list(db["clean_connections"].aggregate(pipeline, allowDiskUse=True))
        
        if results:
            regions = [r["_id"] for r in results if r["_id"]]
            synapses = [r["synapses"] for r in results if r["_id"]]
            connections = [r["connections"] for r in results if r["_id"]]
            gaba_vals = [r.get("gaba", 0) for r in results if r["_id"]]
            glut_vals = [r.get("glut", 0) for r in results if r["_id"]]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=gaba_vals,
                y=glut_vals,
                mode='markers',
                marker=dict(
                    size=[s/500 for s in synapses],
                    color=synapses,
                    colorscale='Plasma',
                    showscale=True,
                    colorbar=dict(
                        title=dict(text="Synapses", font=dict(color='white')),
                        tickfont=dict(color='white')
                    ),
                    line=dict(color='white', width=1.5),
                    opacity=0.8
                ),
                text=regions,
                hovertemplate='<b>%{text}</b><br>GABA: %{x:.4f}<br>Glut: %{y:.4f}<br>Syn: %{marker.color:,}<extra></extra>'
            ))
            
            fig.update_layout(
                title=dict(text="GABA-Glutamate Space", font=dict(size=18, color='white')),
                xaxis_title="GABA",
                yaxis_title="Glutamate",
                height=550,
                plot_bgcolor='#0e1117',
                paper_bgcolor='#0e1117',
                font=dict(color='white'),
                xaxis=dict(gridcolor='#333', color='white'),
                yaxis=dict(gridcolor='#333', color='white')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig2 = go.Figure(go.Bar(
                    x=regions[:10],
                    y=synapses[:10],
                    marker=dict(color=synapses[:10], colorscale='Viridis'),
                    text=[f"{s:,}" for s in synapses[:10]],
                    textposition='outside'
                ))
                
                fig2.update_layout(
                    title="Top 10 Regions by Synapses",
                    height=350,
                    plot_bgcolor='#0e1117',
                    paper_bgcolor='#0e1117',
                    font=dict(color='white'),
                    xaxis=dict(color='white', tickangle=-45),
                    yaxis=dict(gridcolor='#333', color='white')
                )
                
                st.plotly_chart(fig2, use_container_width=True)
            
            with col2:
                fig3 = go.Figure(go.Scatter(
                    x=connections[:10],
                    y=synapses[:10],
                    mode='markers+text',
                    marker=dict(size=15, color='#00cc96'),
                    text=regions[:10],
                    textposition='top center'
                ))
                
                fig3.update_layout(
                    title="Connections vs Synapses",
                    xaxis_title="Connections",
                    yaxis_title="Synapses",
                    height=350,
                    plot_bgcolor='#0e1117',
                    paper_bgcolor='#0e1117',
                    font=dict(color='white'),
                    xaxis=dict(gridcolor='#333', color='white'),
                    yaxis=dict(gridcolor='#333', color='white')
                )
                
                st.plotly_chart(fig3, use_container_width=True)
    
    with tab3:
        sankey_data = {
            "nodes": [
                {"label": f"Raw ({raw_n/1e6:.1f}M)"},
                {"label": f"Clean ({clean_n/1e6:.1f}M)"},
                {"label": f"Region ({region_n})"},
                {"label": f"Tensor ({tensor_n})"},
                {"label": f"Neo4j ({neo4j_nodes})"}
            ],
            "links": [
                {"source": 0, "target": 1, "value": clean_n},
                {"source": 1, "target": 2, "value": region_n},
                {"source": 1, "target": 3, "value": tensor_n},
                {"source": 3, "target": 4, "value": neo4j_nodes}
            ]
        }
        
        fig = go.Figure(go.Sankey(
            node=dict(
                label=[n["label"] for n in sankey_data["nodes"]],
                color=["#636efa", "#00cc96", "#ef553b", "#ffa15a", "#ab63fa"],
                pad=20,
                thickness=30
            ),
            link=dict(
                source=[l["source"] for l in sankey_data["links"]],
                target=[l["target"] for l in sankey_data["links"]],
                value=[l["value"] for l in sankey_data["links"]],
                color="rgba(100,100,100,0.3)"
            )
        ))
        
        fig.update_layout(
            title=dict(text="Pipeline Data Flow", font=dict(size=20, color='white')),
            height=500,
            plot_bgcolor='#0e1117',
            paper_bgcolor='#0e1117',
            font=dict(color='white', size=14)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        retention_data = pl.DataFrame([
            {"Layer": "Raw→Clean", "Retention": f"{retention:.1f}%", "Dropped": f"{raw_n-clean_n:,}"},
            {"Layer": "Clean→Region", "Retention": "Agg", "Output": f"{region_n:,}"},
            {"Layer": "Clean→Tensor", "Retention": "Transform", "Output": f"{tensor_n:,}"},
            {"Layer": "Tensor→Neo4j", "Retention": "Projection", "Output": f"{neo4j_nodes:,}"}
        ])
        
        st.dataframe(retention_data, use_container_width=True, hide_index=True)
    
    st.success("Pipeline operational")

def render_tensor_analytics():
    st.header("Tensor Analytics")
    
    client, ok = get_mongo_client()
    if not ok or not client:
        st.error("MongoDB offline")
        return
    
    db = client["connectome"]
    tensor = db["gold_tensor"]
    
    tensor_n = tensor.estimated_document_count()
    
    if tensor_n == 0:
        st.warning("gold_tensor empty")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Tensor Entries", f"{tensor_n:,}")
    
    pipeline_dims = [
        {"$limit": 1},
        {"$project": {"coords": {"$objectToArray": "$coords"}}}
    ]
    
    sample = list(tensor.aggregate(pipeline_dims))
    if sample and "coords" in sample[0]:
        coord_dims = len(sample[0]["coords"])
        with col2:
            st.metric("Coord Dimensions", coord_dims)
    
    pipeline_metrics = [
        {"$limit": 1},
        {"$project": {"metrics": {"$cond": [{"$isArray": "$metrics"}, "$metrics", {"$objectToArray": "$metrics"}]}}}
    ]
    
    metrics_sample = list(tensor.aggregate(pipeline_metrics))
    metric_count = 0
    
    if metrics_sample:
        m = metrics_sample[0].get("metrics")
        if isinstance(m, list):
            metric_count = len(m)
        elif isinstance(m, dict):
            metric_count = len(m)
    
    with col3:
        st.metric("Metric Channels", metric_count if metric_count > 0 else "Scalar")
    
    stats = db.command(SON([("collStats", "gold_tensor"), ("scale", 1024*1024)]))
    storage = stats.get("storageSize", 0)
    
    with col4:
        st.metric("Storage", f"{storage:.2f} MB")
    
    st.divider()
    
    tab1, tab2, tab3 = st.tabs(["Distribution", "Projection", "Slice"])
    
    with tab1:
        pipeline = [
            {"$match": {"value": {"$ne": None, "$exists": True}}},
            {"$limit": 5000},
            {"$project": {"value": 1}}
        ]
        
        results = list(tensor.aggregate(pipeline, allowDiskUse=True))
        
        if results:
            values = [r["value"] for r in results if isinstance(r.get("value"), (int, float))]
            
            if values:
                df = pl.DataFrame({"value": values})
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    fig = go.Figure()
                    
                    fig.add_trace(go.Histogram(
                        x=values,
                        nbinsx=50,
                        marker=dict(
                            color=values,
                            colorscale='Viridis',
                            line=dict(color='white', width=1)
                        )
                    ))
                    
                    fig.update_layout(
                        title=dict(text="Value Distribution", font=dict(size=18, color='white')),
                        xaxis_title="Value",
                        yaxis_title="Frequency",
                        height=400,
                        plot_bgcolor='#0e1117',
                        paper_bgcolor='#0e1117',
                        font=dict(color='white'),
                        xaxis=dict(gridcolor='#333'),
                        yaxis=dict(gridcolor='#333')
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    stats_df = df.describe()
                    st.dataframe(stats_df, use_container_width=True)
                
                st.divider()
                
                fig2 = go.Figure()
                
                fig2.add_trace(go.Box(
                    y=values,
                    marker=dict(color='#00cc96'),
                    name="Tensor Values"
                ))
                
                fig2.update_layout(
                    title="Value Distribution (Box Plot)",
                    height=300,
                    plot_bgcolor='#0e1117',
                    paper_bgcolor='#0e1117',
                    font=dict(color='white'),
                    yaxis=dict(gridcolor='#333', color='white')
                )
                
                st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        pipeline_coords = [
            {"$limit": 2000},
            {"$project": {
                "src": "$coords.src",
                "tgt": "$coords.tgt",
                "metric": "$coords.metric",
                "value": 1
            }},
            {"$match": {"src": {"$ne": None}, "tgt": {"$ne": None}}}
        ]
        
        results = list(tensor.aggregate(pipeline_coords, allowDiskUse=True))
        
        if results:
            src_counts = {}
            tgt_counts = {}
            
            for r in results:
                src = r.get("src")
                tgt = r.get("tgt")
                if src:
                    src_counts[src] = src_counts.get(src, 0) + 1
                if tgt:
                    tgt_counts[tgt] = tgt_counts.get(tgt, 0) + 1
            
            top_src = sorted(src_counts.items(), key=lambda x: x[1], reverse=True)[:15]
            top_tgt = sorted(tgt_counts.items(), key=lambda x: x[1], reverse=True)[:15]
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure(go.Bar(
                    x=[s[0] for s in top_src],
                    y=[s[1] for s in top_src],
                    marker=dict(color=[s[1] for s in top_src], colorscale='Blues'),
                    text=[f"{s[1]:,}" for s in top_src],
                    textposition='outside'
                ))
                
                fig.update_layout(
                    title="Top Source Regions",
                    height=350,
                    plot_bgcolor='#0e1117',
                    paper_bgcolor='#0e1117',
                    font=dict(color='white'),
                    xaxis=dict(tickangle=-45, color='white'),
                    yaxis=dict(gridcolor='#333', color='white')
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = go.Figure(go.Bar(
                    x=[t[0] for t in top_tgt],
                    y=[t[1] for t in top_tgt],
                    marker=dict(color=[t[1] for t in top_tgt], colorscale='Reds'),
                    text=[f"{t[1]:,}" for t in top_tgt],
                    textposition='outside'
                ))
                
                fig.update_layout(
                    title="Top Target Regions",
                    height=350,
                    plot_bgcolor='#0e1117',
                    paper_bgcolor='#0e1117',
                    font=dict(color='white'),
                    xaxis=dict(tickangle=-45, color='white'),
                    yaxis=dict(gridcolor='#333', color='white')
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            
            src_vals = list(src_counts.values())
            tgt_vals = list(tgt_counts.values())
            
            fig3 = go.Figure()
            
            fig3.add_trace(go.Scatter(
                x=src_vals,
                y=tgt_vals,
                mode='markers',
                marker=dict(
                    size=8,
                    color=src_vals,
                    colorscale='Plasma',
                    showscale=True,
                    colorbar=dict(title=dict(text="Frequency", font=dict(color='white')), tickfont=dict(color='white'))
                )
            ))
            
            fig3.update_layout(
                title="Source vs Target Cardinality",
                xaxis_title="Source Frequency",
                yaxis_title="Target Frequency",
                height=400,
                plot_bgcolor='#0e1117',
                paper_bgcolor='#0e1117',
                font=dict(color='white'),
                xaxis=dict(gridcolor='#333', color='white', type='log'),
                yaxis=dict(gridcolor='#333', color='white', type='log')
            )
            
            st.plotly_chart(fig3, use_container_width=True)
    
    with tab3:
        limit = st.slider("Sample Size", 10, 100, 20)
        
        samples = list(tensor.find({}, {"_id": 0}).limit(limit))
        
        if samples:
            df = pl.DataFrame(samples)
            st.dataframe(df, use_container_width=True, height=400)
            
            with st.expander("Raw JSON"):
                st.json(samples[:5])
    
    st.success("Tensor layer analyzed")

def project_tensor_to_neo4j(top_k=300):
    mongo = MongoClient("mongodb://mongo-primary:27017/?replicaSet=rs0")
    db = mongo["connectome"]
    tensor = db["gold_tensor"]

    driver = GraphDatabase.driver(
        "bolt://neo4j:7687",
        auth=("neo4j", "password")
    )

    pipeline = [
        {
            "$match": {
                "coords.src": {"$ne": None},
                "coords.tgt": {"$ne": None}
            }
        },
        {
            "$project": {
                "src": "$coords.src",
                "tgt": "$coords.tgt",
                "value": {
                    "$cond": [
                        {"$or": [{"$eq": ["$value", None]}, {"$ne": ["$value", "$value"]}]},
                        0,
                        "$value"
                    ]
                },
                "support": {"$ifNull": ["$support", 1]}
            }
        },
        {
            "$match": {
                "$expr": {"$ne": ["$src", "$tgt"]}
            }
        },
        {
            "$group": {
                "_id": {"s": "$src", "t": "$tgt"},
                "w": {
                    "$sum": {
                        "$multiply": [
                            {"$abs": "$value"},
                            {"$ln": {"$add": ["$support", 1]}}
                        ]
                    }
                }
            }
        },
        {"$sort": {"w": -1}},
        {"$limit": top_k}
    ]

    edges = list(tensor.aggregate(pipeline, allowDiskUse=True))

    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")

        for e in edges:
            session.run(
                """
                MERGE (a:Neuron {name:$s})
                MERGE (b:Neuron {name:$t})
                MERGE (a)-[r:SYNAPSE]->(b)
                SET r.weight = $w
                """,
                s=e["_id"]["s"],
                t=e["_id"]["t"],
                w=float(e["w"])
            )

    driver.close()
def render_neo4j_graph():
    st.header("Graph Connectomics")
    
    driver, ok = get_neo4j_driver()
    if not ok or not driver:
        st.error("Neo4j offline")
        return
    
    try:
        with driver.session() as session:
            stats = session.run("""
                MATCH (n)
                OPTIONAL MATCH ()-[r]->()
                RETURN count(DISTINCT n) AS nodes, count(r) AS edges
            """).single()
            
            node_count = stats["nodes"]
            rel_count = stats["edges"]
            
            if node_count == 0:
                st.warning("Graph empty. Run: project_tensor_to_neo4j()")
                return
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Nodes", f"{node_count:,}")
            with col2:
                st.metric("Edges", f"{rel_count:,}")
            with col3:
                density = (rel_count / (node_count * (node_count - 1))) if node_count > 1 else 0
                st.metric("Density", f"{density:.6f}")
            with col4:
                avg_degree = (2 * rel_count / node_count) if node_count > 0 else 0
                st.metric("Avg Degree", f"{avg_degree:.2f}")
            
            st.divider()
            
            tab1, tab2, tab3 = st.tabs(["Network", "Statistics", "Topology"])
            
            with tab1:
                k = st.slider("Top-K Connections", 50, 500, 200)
                
                rows = session.run("""
                    MATCH (a)-[r:SYNAPSE]->(b)
                    WHERE r.weight IS NOT NULL
                    RETURN
                        a.name AS source,
                        b.name AS target,
                        r.weight AS weight,
                        r.metric AS metric
                    ORDER BY r.weight DESC
                    LIMIT $k
                """, k=k).data()
                
                if not rows:
                    st.info("No weighted connections")
                    return
                
                nodes_dict = {}
                links = []
                
                for r in rows:
                    s, t = r["source"], r["target"]
                    if not s or not t:
                        continue
                    
                    nodes_dict.setdefault(s, {"id": s, "degree": 0})
                    nodes_dict.setdefault(t, {"id": t, "degree": 0})
                    nodes_dict[s]["degree"] += 1
                    nodes_dict[t]["degree"] += 1
                    
                    links.append({
                        "source": s,
                        "target": t,
                        "weight": float(r["weight"]),
                        "metric": r.get("metric", "")
                    })
                
                graph_data = {"nodes": list(nodes_dict.values()), "links": links}
                
                html_content = """
<!DOCTYPE html>
<html>
<head>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
body { margin:0; background:#0e1117; font-family:monospace; overflow:hidden; }
#container { display:flex; height:700px; }
#graph { flex:1; position:relative; }
#panel {
    width:280px;
    background:#1a1a1a;
    color:#eee;
    padding:16px;
    border-left:2px solid #333;
    overflow-y:auto;
}
#panel h4 { margin:0 0 12px 0; color:#00cc96; font-size:16px; }
.metric { margin:8px 0; padding:10px; background:#262626; border-radius:4px; }
.metric-label { color:#999; font-size:11px; text-transform:uppercase; }
.metric-value { color:#fff; font-size:18px; font-weight:600; margin-top:4px; }
.faded { opacity:0.08 !important; }
.highlighted { stroke:#00cc96 !important; stroke-width:3px !important; }
#controls { position:absolute; top:10px; right:10px; background:#1a1a1a; padding:10px; border-radius:4px; border:1px solid #333; }
#controls button { background:#00cc96; color:#000; border:none; padding:6px 12px; border-radius:3px; cursor:pointer; font-weight:600; font-size:11px; }
#controls button:hover { background:#00e6b8; }
</style>
</head>
<body>

<div id="container">
<div id="graph">
    <div id="controls">
        <button onclick="resetView()">RESET VIEW</button>
    </div>
</div>
<div id="panel">
    <h4>NETWORK EXPLORER</h4>
    <div id="info">
        <div class="metric">
            <div class="metric-label">Nodes</div>
            <div class="metric-value">""" + str(len(nodes_dict)) + """</div>
        </div>
        <div class="metric">
            <div class="metric-label">Edges</div>
            <div class="metric-value">""" + str(len(links)) + """</div>
        </div>
        <div style="margin-top:20px; color:#999; font-size:12px; line-height:1.6;">
            • Click node to explore<br/>
            • Drag to reposition<br/>
            • Size ∝ Degree<br/>
            • Color ∝ Centrality<br/>
            • Edge width ∝ Weight
        </div>
    </div>
</div>
</div>

<script>
const data = """ + json.dumps(graph_data) + """;
const width = document.getElementById("graph").clientWidth;
const height = 700;

const svg = d3.select("#graph")
    .append("svg")
    .attr("width", width)
    .attr("height", height);

const maxDegree = d3.max(data.nodes, d => d.degree) || 1;

const linked = {};
data.links.forEach(l => {
    linked[l.source + "," + l.target] = true;
    linked[l.target + "," + l.source] = true;
});

const isConnected = (a, b) => linked[a + "," + b] || a === b;

const simulation = d3.forceSimulation(data.nodes)
    .force("link", d3.forceLink(data.links).id(d => d.id).distance(100))
    .force("charge", d3.forceManyBody().strength(-400))
    .force("center", d3.forceCenter(width/2, height/2))
    .force("collision", d3.forceCollide().radius(d => 10 + Math.sqrt(d.degree) * 3));

const link = svg.append("g")
    .selectAll("line")
    .data(data.links)
    .enter().append("line")
    .attr("stroke", "#555")
    .attr("stroke-width", d => Math.sqrt(d.weight) * 1.2)
    .attr("opacity", 0.6);

const node = svg.append("g")
    .selectAll("circle")
    .data(data.nodes)
    .enter().append("circle")
    .attr("r", d => 6 + Math.sqrt(d.degree) * 2.5)
    .attr("fill", d => d3.interpolatePlasma(d.degree / maxDegree))
    .attr("stroke", "#fff")
    .attr("stroke-width", 2)
    .style("cursor", "pointer")
    .call(d3.drag()
        .on("start", (e, d) => { if(!e.active) simulation.alphaTarget(0.3).restart(); d.fx=d.x; d.fy=d.y; })
        .on("drag", (e, d) => { d.fx=e.x; d.fy=e.y; })
        .on("end", (e, d) => { if(!e.active) simulation.alphaTarget(0); d.fx=d.fy=null; })
    )
    .on("click", (e, d) => {
        e.stopPropagation();
        selectNode(d);
    });

function selectNode(d) {
    node.classed("faded", n => n.id !== d.id && !isConnected(n.id, d.id));
    link.classed("faded", l => l.source.id !== d.id && l.target.id !== d.id)
        .classed("highlighted", l => l.source.id === d.id || l.target.id === d.id);
    
    const outgoing = data.links.filter(l => l.source.id === d.id).length;
    const incoming = data.links.filter(l => l.target.id === d.id).length;
    
    const neighbors = data.links
        .filter(l => l.source.id === d.id || l.target.id === d.id)
        .map(l => ({
            id: l.source.id === d.id ? l.target.id : l.source.id,
            weight: l.weight,
            metric: l.metric
        }))
        .sort((a, b) => b.weight - a.weight)
        .slice(0, 10);
    
    const neighborHTML = neighbors.map(n => 
        `<div style="padding:4px 0; border-bottom:1px solid #333;">
            <div style="color:#00cc96; font-size:11px;">${n.id.substring(0,16)}</div>
            <div style="color:#999; font-size:10px;">w:${n.weight.toFixed(4)} | ${n.metric}</div>
        </div>`
    ).join('');
    
    document.getElementById("info").innerHTML = `
        <div class="metric">
            <div class="metric-label">Node ID</div>
            <div class="metric-value" style="font-size:13px; word-break:break-all;">${d.id}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Total Degree</div>
            <div class="metric-value">${d.degree}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Out / In</div>
            <div class="metric-value" style="font-size:15px;">${outgoing} / ${incoming}</div>
        </div>
        <div style="margin-top:16px; color:#999; font-size:12px; margin-bottom:8px;">TOP CONNECTIONS</div>
        <div style="max-height:300px; overflow-y:auto;">${neighborHTML}</div>
    `;
}

function resetView() {
    node.classed("faded", false);
    link.classed("faded", false).classed("highlighted", false);
    document.getElementById("info").innerHTML = `
        <div class="metric">
            <div class="metric-label">Nodes</div>
            <div class="metric-value">""" + str(len(nodes_dict)) + """</div>
        </div>
        <div class="metric">
            <div class="metric-label">Edges</div>
            <div class="metric-value">""" + str(len(links)) + """</div>
        </div>
        <div style="margin-top:20px; color:#999; font-size:12px; line-height:1.6;">
            • Click node to explore<br/>
            • Drag to reposition<br/>
            • Size ∝ Degree<br/>
            • Color ∝ Centrality<br/>
            • Edge width ∝ Weight
        </div>
    `;
}

svg.on("click", (e) => {
    if(e.target.tagName === "svg") {
        resetView();
    }
});

simulation.on("tick", () => {
    link
        .attr("x1", d => d.source.x)
        .attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x)
        .attr("y2", d => d.target.y);
    
    node
        .attr("cx", d => d.x)
        .attr("cy", d => d.y);
});
</script>
</body>
</html>
                """
                
                components.html(html_content, height=720)
            
            with tab2:
                degree_dist = session.run("""
                    MATCH (n:Neuron)
                    OPTIONAL MATCH (n)-[r_out:SYNAPSE]->()
                    OPTIONAL MATCH (n)<-[r_in:SYNAPSE]-()
                    WITH n, count(DISTINCT r_out) as out_degree, count(DISTINCT r_in) as in_degree
                    RETURN 
                        n.name as node,
                        out_degree,
                        in_degree,
                        out_degree + in_degree as total_degree
                    ORDER BY total_degree DESC
                    LIMIT 30
                """).data()
                
                if degree_dist:
                    nodes = [d["node"] for d in degree_dist]
                    out_deg = [d["out_degree"] for d in degree_dist]
                    in_deg = [d["in_degree"] for d in degree_dist]
                    total_deg = [d["total_degree"] for d in degree_dist]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = go.Figure()
                        
                        fig.add_trace(go.Bar(
                            x=nodes[:15],
                            y=out_deg[:15],
                            name='Out',
                            marker=dict(color='#636efa')
                        ))
                        
                        fig.add_trace(go.Bar(
                            x=nodes[:15],
                            y=in_deg[:15],
                            name='In',
                            marker=dict(color='#ef553b')
                        ))
                        
                        fig.update_layout(
                            title="Degree Distribution",
                            barmode='group',
                            height=400,
                            plot_bgcolor='#0e1117',
                            paper_bgcolor='#0e1117',
                            font=dict(color='white'),
                            xaxis=dict(tickangle=-45, color='white'),
                            yaxis=dict(gridcolor='#333', color='white'),
                            legend=dict(font=dict(color='white'))
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig2 = go.Figure()
                        
                        fig2.add_trace(go.Scatter(
                            x=out_deg,
                            y=in_deg,
                            mode='markers',
                            marker=dict(
                                size=12,
                                color=total_deg,
                                colorscale='Viridis',
                                showscale=True,
                                colorbar=dict(title=dict(text="Total", font=dict(color='white')), tickfont=dict(color='white'))
                            ),
                            text=nodes,
                            hovertemplate='<b>%{text}</b><br>Out: %{x}<br>In: %{y}<extra></extra>'
                        ))
                        
                        fig2.update_layout(
                            title="In vs Out Degree",
                            xaxis_title="Out",
                            yaxis_title="In",
                            height=400,
                            plot_bgcolor='#0e1117',
                            paper_bgcolor='#0e1117',
                            font=dict(color='white'),
                            xaxis=dict(gridcolor='#333', color='white'),
                            yaxis=dict(gridcolor='#333', color='white')
                        )
                        
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    st.divider()
                    
                    df = pl.DataFrame(degree_dist[:20])
                    st.dataframe(df, use_container_width=True, hide_index=True)
            
            with tab3:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    triangles = session.run("""
                        MATCH (a:Neuron)-[:SYNAPSE]->(b:Neuron)-[:SYNAPSE]->(c:Neuron)-[:SYNAPSE]->(a)
                        RETURN count(*) as triangles
                    """).single()
                    
                    triangle_count = triangles["triangles"] if triangles else 0
                    st.metric("Triangles", f"{triangle_count:,}")
                
                with col2:
                    clustering = (3 * triangle_count / rel_count) if rel_count > 0 else 0
                    st.metric("Clustering", f"{clustering:.4f}")
                
                with col3:
                    try:
                        comp_result = session.run("""
                            MATCH (n:Neuron)
                            WITH collect(id(n)) as node_ids
                            UNWIND node_ids as node_id
                            MATCH path = (start:Neuron)-[:SYNAPSE*..5]-(end:Neuron)
                            WHERE id(start) = node_id
                            WITH node_id, collect(DISTINCT id(end)) as connected
                            RETURN count(DISTINCT node_id) as components
                        """).single()
                        
                        comp_count = comp_result["components"] if comp_result else 1
                    except:
                        comp_count = 1
                    
                    st.metric("Reachability", f"{comp_count:,}")
                
                st.divider()
                
                weight_dist = session.run("""
                    MATCH ()-[r:SYNAPSE]->()
                    WHERE r.weight IS NOT NULL
                    RETURN r.weight as weight
                    ORDER BY r.weight DESC
                    LIMIT 1000
                """).data()
                
                if weight_dist:
                    weights = [w["weight"] for w in weight_dist if w["weight"]]
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Histogram(
                        x=weights,
                        nbinsx=50,
                        marker=dict(color='#00cc96', line=dict(color='white', width=1))
                    ))
                    
                    fig.update_layout(
                        title="Edge Weight Distribution",
                        xaxis_title="Weight",
                        yaxis_title="Frequency",
                        height=350,
                        plot_bgcolor='#0e1117',
                        paper_bgcolor='#0e1117',
                        font=dict(color='white'),
                        xaxis=dict(gridcolor='#333', color='white'),
                        yaxis=dict(gridcolor='#333', color='white')
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        st.success("Graph validated")
    
    except Exception as e:
        st.error(f"Neo4j error: {e}")
        st.exception(e)

def main():
    client, mongo_ok = get_mongo_client()
    driver, neo4j_ok = get_neo4j_driver()
    
    st.sidebar.title("Navigation")
    
    page = st.sidebar.radio(
        "Select View",
        [
            "System Overview",
            "Ingest Status",
            "Clean Layer",
            "Gold Analytics",
            "Pipelines",
            "Tensor Analytics",
            "Neo4j Graph"
        ]
    )
    
    st.sidebar.divider()
    
    st.sidebar.subheader("Connection Status")
    st.sidebar.write(f"MongoDB: {'CONNECTED' if mongo_ok else 'OFFLINE'}")
    st.sidebar.write(f"Neo4j: {'CONNECTED' if neo4j_ok else 'OFFLINE'}")
    
    if page == "System Overview":
        render_system_overview(mongo_ok, neo4j_ok)
    elif page == "Ingest Status":
        render_ingest_status()
    elif page == "Clean Layer":
        render_clean_layer()
    elif page == "Gold Analytics":
        render_gold_analytics()
    elif page == "Pipelines":
        render_pipelines()
    elif page == "Neo4j Graph":
        render_neo4j_graph()
    elif page == "Tensor Analytics":
        render_tensor_analytics()


if __name__ == "__main__":
    main()