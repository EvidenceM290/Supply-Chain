import networkx as nx
import numpy as np
import random
import faiss
import openai
from collections import defaultdict
import plotly.graph_objects as go
from geopy.geocoders import Nominatim
import pandas as pd
import gradio as gr
import os
from gmplot import gmplot
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
import string
import requests
import re

random.seed(42)  # Ensures reproducibility

# --- Configuration ---
OPENAI_API_KEY = "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"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY
geolocator = Nominatim(user_agent="logigraph")
GOOGLE_MAPS_API_KEY = "AIzaSyBO3d4h8aJA_KBieC27fHS3AJzzXprNZto"
OPENWEATHERMAP_API_KEY = "your_openweathermap_api_key"

# --- Create graph ---
G = nx.MultiDiGraph()
entity_types = ["Warehouse", "Distribution Center", "Retail Store"]
locations = [
    "New York, USA", "London, UK", "Tokyo, Japan", "Berlin, Germany", "Toronto, Canada",
    "Sydney, Australia", "Paris, France", "Dubai, UAE", "São Paulo, Brazil", "Mumbai, India",
    "Cape Town, South Africa", "Beijing, China", "Moscow, Russia", "Istanbul, Turkey", "Bangkok, Thailand",
    "Seoul, South Korea", "Mexico City, Mexico", "Rome, Italy", "Madrid, Spain", "Lima, Peru"
]
modes = ["Air", "Road", "Sea", "Rail"]

# Continent mapping for locations
location_continents = {
    "New York, USA": "North America",
    "London, UK": "Europe", 
    "Tokyo, Japan": "Asia",
    "Berlin, Germany": "Europe",
    "Toronto, Canada": "North America",
    "Sydney, Australia": "Australia",
    "Paris, France": "Europe",
    "Dubai, UAE": "Asia",
    "São Paulo, Brazil": "South America",
    "Mumbai, India": "Asia",
    "Cape Town, South Africa": "Africa",
    "Beijing, China": "Asia",
    "Moscow, Russia": "Europe",
    "Istanbul, Turkey": "Europe",
    "Bangkok, Thailand": "Asia",
    "Seoul, South Korea": "Asia",
    "Mexico City, Mexico": "North America",
    "Rome, Italy": "Europe",
    "Madrid, Spain": "Europe",
    "Lima, Peru": "South America"
}

node_coords = {}
nodes = []
risk_levels = ['low', 'medium', 'high']
used_locations = set()  # Track used locations to avoid duplicates

for i in range(20):  # Changed from 25 to 50 nodes
    node_id = str(i)  # Use just the number as node ID
    
    # Keep trying until we get a unique location
    location = None
    attempts = 0
    while location is None or location in used_locations:
        location = random.choice(locations)
        attempts += 1
        if attempts > 100:  # Prevent infinite loop
            break
    
    if location is None:
        # If we can't find a unique location, use the first available one
        available_locations = [loc for loc in locations if loc not in used_locations]
        if available_locations:
            location = available_locations[0]
        else:
            location = random.choice(locations)  # Fallback
    
    used_locations.add(location)
    continent = location_continents[location]
    
    # Get coordinates using geolocation
    try:
        geo = geolocator.geocode(location, timeout=10)
        print(f"Geocoding '{location}': {geo}")
        if geo:
            lat, lon = geo.latitude, geo.longitude
            node_coords[node_id] = (lat, lon)
        else:
            print(f"Could not geocode '{location}', using fallback coordinates")
            # Fallback coordinates for common locations
            fallback_coords = {
                "New York, USA": (40.7128, -74.0060),
                "London, UK": (51.5074, -0.1278),
                "Tokyo, Japan": (35.6895, 139.6917),
                "Berlin, Germany": (52.52, 13.4050),
                "Toronto, Canada": (43.65107, -79.347015),
                "Sydney, Australia": (-33.8688, 151.2093),
                "Paris, France": (48.8566, 2.3522),
                "Dubai, UAE": (25.2048, 55.2708),
                "São Paulo, Brazil": (-23.5505, -46.6333),
                "Mumbai, India": (19.0760, 72.8777),
                "Cape Town, South Africa": (-33.9249, 18.4241),
                "Beijing, China": (39.9042, 116.4074),
                "Moscow, Russia": (55.7558, 37.6173),
                "Istanbul, Turkey": (41.0082, 28.9784),
                "Bangkok, Thailand": (13.7563, 100.5018),
                "Seoul, South Korea": (37.5665, 126.9780),
                "Mexico City, Mexico": (19.4326, -99.1332),
                "Rome, Italy": (41.9028, 12.4964),
                "Madrid, Spain": (40.4168, -3.7038),
                "Lima, Peru": (-12.0464, -77.0428)
            }
            lat, lon = fallback_coords.get(location, (0, 0))
            node_coords[node_id] = (lat, lon)
    except Exception as e:
        print(f"Geocoding error for '{location}': {e}")
        # Use fallback coordinates
        fallback_coords = {
            "New York, USA": (40.7128, -74.0060),
            "London, UK": (51.5074, -0.1278),
            "Tokyo, Japan": (35.6895, 139.6917),
            "Berlin, Germany": (52.52, 13.4050),
            "Toronto, Canada": (43.65107, -79.347015),
            "Sydney, Australia": (-33.8688, 151.2093),
            "Paris, France": (48.8566, 2.3522),
            "Dubai, UAE": (25.2048, 55.2708),
            "São Paulo, Brazil": (-23.5505, -46.6333),
            "Mumbai, India": (19.0760, 72.8777),
            "Cape Town, South Africa": (-33.9249, 18.4241),
            "Beijing, China": (39.9042, 116.4074),
            "Moscow, Russia": (55.7558, 37.6173),
            "Istanbul, Turkey": (41.0082, 28.9784),
            "Bangkok, Thailand": (13.7563, 100.5018),
            "Seoul, South Korea": (37.5665, 126.9780),
            "Mexico City, Mexico": (19.4326, -99.1332),
            "Rome, Italy": (41.9028, 12.4964),
            "Madrid, Spain": (40.4168, -3.7038),
            "Lima, Peru": (-12.0464, -77.0428)
        }
        lat, lon = fallback_coords.get(location, (0, 0))
        node_coords[node_id] = (lat, lon)
    
    risk_level = random.choice(risk_levels)  # Assign risk level to each node
    node_attr = {
        "type": random.choice(entity_types),
        "location": location,
        "lat": lat,
        "lon": lon,
        "continent": continent,
        "storage_capacity_tons": random.randint(1000, 100000),
        "operational_status": random.choice(["active", "maintenance", "inactive"]),
        "inventory_level": random.randint(100, 90000),
        "risk_level": risk_level  # Add risk level to node attributes
    }
    G.add_node(node_id, **node_attr)
    nodes.append(node_id)

# Print node assignments for user reference
for node_id in nodes:
    print(f"{node_id}: {G.nodes[node_id]['location']}")

# First, connect all nodes in a chain to ensure connectivity
for i in range(len(nodes) - 1):
    u, v = nodes[i], nodes[i + 1]
    edge_attr = {
        "mode": random.choice(modes),
        "time_days": random.randint(1, 15),
        "cost_usd": random.randint(500, 15000),
        "risk": random.choice(["low", "medium", "high"]),
        "capacity_tons": random.randint(50, 5000),
        "status": random.choice(["on_time", "delayed", "blocked"]),
        "delay_probability": round(random.uniform(0.01, 0.9), 2)
    }
    G.add_edge(u, v, **edge_attr)

# Add random extra edges until total number of edges is 30
max_edges = 30
while G.number_of_edges() < max_edges:
    u, v = random.sample(nodes, 2)
    if u != v and not G.has_edge(u, v):
        edge_attr = {
            "mode": random.choice(modes),
            "time_days": random.randint(1, 15),
            "cost_usd": random.randint(500, 15000),
            "risk": random.choice(["low", "medium", "high"]),
            "capacity_tons": random.randint(50, 5000),
            "status": random.choice(["on_time", "delayed", "blocked"]),
            "delay_probability": round(random.uniform(0.01, 0.9), 2)
        }
        G.add_edge(u, v, **edge_attr)

# --- Embeddings ---
disruptions = ["Port closure", "Weather delay", "System outage"]
embedding_vectors = []
for text in disruptions:
    response = openai.Embedding.create(model="text-embedding-ada-002", input=text)
    embedding_vectors.append(response['data'][0]['embedding'])
embedding_vectors = np.array(embedding_vectors).astype('float32')
index = faiss.IndexFlatL2(len(embedding_vectors[0]))
index.add(embedding_vectors)

def retrieve_similar_disruptions(query_embedding):
    D, I = index.search(np.array([query_embedding], dtype='float32'), k=2)
    return [disruptions[i] for i in I[0]]

def get_lowest_risk_route(source, target):
    risk_map = {"low": 1, "medium": 2, "high": 3}
    try:
        path = nx.shortest_path(G, source=source, target=target, weight=lambda u, v, d: risk_map[d['risk']])
        total_risk = sum(risk_map[G[u][v]['risk']] for u, v in zip(path[:-1], path[1:]))
        return path, total_risk
    except nx.NetworkXNoPath:
        return None, float('inf')

def get_all_routes_with_status(source, target):
    all_routes = list(nx.all_simple_paths(G, source=source, target=target))
    route_infos = []
    risk_map = {"low": 1, "medium": 2, "high": 3}
    for route in all_routes:
        # For MultiDiGraph, you need to consider all edge keys between each pair
        all_edge_combinations = [[]]
        for u, v in zip(route[:-1], route[1:]):
            edge_data = G.get_edge_data(u, v)
            if edge_data:
                # edge_data is a dict of {key: attr_dict}
                all_edge_combinations = [
                    prev + [(u, v, k, edge_data[k])]
                    for prev in all_edge_combinations
                    for k in edge_data
                ]
        for edge_combo in all_edge_combinations:
            total_cost = 0
            total_risk = 0
            total_delay = 0
            total_time = 0
            risk_values = []
            blocked = False
            # Only blocked status matters; ignore node operational status
            for (u, v, k, edge) in edge_combo:
                total_cost += float(edge.get('cost_usd', 0) or 0)
                risk_val = risk_map.get(edge.get('risk', 'low'), 1)
                total_risk += risk_val
                risk_values.append(risk_val)
                total_delay += float(edge.get('delay_probability', 0) or 0)
                total_time += float(edge.get('time_days', 0) or 0)
                if edge.get('status') == 'blocked':
                    blocked = True
            # Compute qualitative risk
            avg_risk = sum(risk_values) / len(risk_values) if risk_values else 1
            if avg_risk < 1.5:
                qualitative_risk = 'Low'
            elif avg_risk < 2.5:
                qualitative_risk = 'Medium'
            else:
                qualitative_risk = 'High'
            route_infos.append({
                'route': route,
                'edges': edge_combo,
                'total_cost': total_cost,
                'total_risk': qualitative_risk,  # keep for display
                'total_risk_numeric': avg_risk,   # add for computation
                'total_delay': total_delay,
                'total_time': total_time,
                'blocked': blocked,
                'inactive': False  # Always False now
            })
    return route_infos

def select_optimal_route(route_infos):
    # Only consider non-blocked and active routes
    valid_routes = [r for r in route_infos if not r['blocked']]
    if not valid_routes:
        return None
    # Optimize: lowest (cost + risk*1000 + delay*1000) for demo
    optimal = min(valid_routes, key=lambda r: r['total_cost'] + r['total_risk_numeric']*1000 + r['total_delay']*1000)
    return optimal

def draw_all_routes_on_map(route_infos, optimal_route):
    if not route_infos:
        print("No route_infos to plot.")
        return None

    # Gather all lat/lon for all nodes in all routes
    all_lats = []
    all_lons = []
    for info in route_infos:
        for n in info['route']:
            lat, lon = node_coords[n]
            if (lat, lon) != (0, 0):
                all_lats.append(lat)
                all_lons.append(lon)
    if not all_lats or not all_lons:
        print("No valid coordinates to plot.")
        return None

    # Center map on the mean of all coordinates, zoom out to fit
    center_lat = sum(all_lats) / len(all_lats)
    center_lon = sum(all_lons) / len(all_lons)
    gmap = gmplot.GoogleMapPlotter(center_lat, center_lon, 2, apikey=GOOGLE_MAPS_API_KEY)

    print(f"Total routes to plot: {len(route_infos)}")
    for idx, info in enumerate(route_infos):
        print(f"Route {idx}: {info['route']} | Blocked: {info['blocked']}")
        route = info['route']
        lats = [node_coords[n][0] for n in route if node_coords[n] != (0, 0)]
        lons = [node_coords[n][1] for n in route if node_coords[n] != (0, 0)]
        if not lats or not lons:
            print("  Skipping route due to missing coordinates.")
            continue
        color = 'blue'
        if info['blocked']:
            color = 'black'
        elif optimal_route and route == optimal_route['route']:
            color = 'green'
        gmap.plot(lats, lons, color, edge_width=4)
        gmap.scatter(lats, lons, color=color, size=40, marker=True)

    # Add node markers with labels
    for node, (lat, lon) in node_coords.items():
        if (lat, lon) != (0, 0):
            gmap.marker(lat, lon, title=node)

    map_file = "route_map.html"
    gmap.draw(map_file)

    # Add a legend to the HTML file BEFORE screenshot
    legend_html = """
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background: white; padding: 10px; border: 2px solid black;">
      <b>Legend</b><br>
      <span style="color:green;">&#9632;</span> Optimal Route<br>
      <span style="color:blue;">&#9632;</span> Valid Route<br>
      <span style="color:black;">&#9632;</span> Blocked Route<br>
    </div>
    """
    with open(map_file, "a") as f:
        f.write(legend_html)

    screenshot_file = "route_map.png"
    print(f"Drawing map to {map_file}, will screenshot to {screenshot_file}")
    # Increase sleep time to ensure rendering
    screenshot_map(map_file, screenshot_file, wait_time=4)
    print(f"Screenshot saved: {os.path.exists(screenshot_file)}")
    return screenshot_file

def describe_routes_for_llm(route_infos, optimal_route):
    desc = []
    for info in route_infos:
        route = info['route']
        status = 'optimal' if optimal_route and info['route'] == optimal_route['route'] else (
            'blocked' if info['blocked'] else 'valid')
        desc.append(f"Route: {' → '.join([G.nodes[n]['location'] for n in route])} | Status: {status.capitalize()}")
        # Segment-by-segment breakdown
        for idx in range(len(route) - 1):
            u, v = route[idx], route[idx + 1]
            # For MultiDiGraph, get all edge data and pick the first (or best)
            edge_data = G.get_edge_data(u, v)
            if edge_data:
                # Pick the first edge (or you could pick the one with lowest risk/cost)
                edge = list(edge_data.values())[0]
                mode = edge.get('mode', '-')
                cost = edge.get('cost_usd', '-')
                risk = edge.get('risk', '-')
                delay = edge.get('delay_probability', '-')
                status_seg = edge.get('status', '-')
                desc.append(f"  {G.nodes[u]['location']} → {G.nodes[v]['location']}: Mode={mode}, Cost=${cost:,}, Risk={risk.capitalize()}, Delay={int(float(delay)*100)}%, Status={status_seg.replace('_', ' ').capitalize()}")
    if optimal_route:
        desc.append(f"\nOptimal route is: {' → '.join([G.nodes[n]['location'] for n in optimal_route['route']])}")
    return '\n'.join(desc)

def find_node_by_location(query):
    def normalize(s):
        return re.sub(r'[^a-z0-9]', '', s.lower())
    norm_query = normalize(query)
    # First try exact match
    for node, attr in G.nodes(data=True):
        loc = attr.get('location', '')
        if normalize(loc) == norm_query:
            return node
    # Then try substring match
    for node, attr in G.nodes(data=True):
        loc = attr.get('location', '')
        if norm_query in normalize(loc) or normalize(loc) in norm_query:
            return node
    # Try country-only match (last word in location)
    for node, attr in G.nodes(data=True):
        loc = attr.get('location', '')
        country = loc.split(',')[-1].strip().lower()
        if norm_query in normalize(country):
            return node
    return None

def query_logigraph_with_agents(question):
    # --- Semantic search for question type ---
    question_types = [
        ("route_status", "Is the route between X and Y blocked or active?"),
        ("route_query", "What is the route from X to Y?"),
        ("operational_status", "Is the location X active or inactive?"),
        ("operational_status", "What is the operational status of X?"),
        ("operational_status", "What is the location status of X?"),
        ("operational_status", "Location status of X?"),
        ("operational_status", "Status of X?"),
        ("general", "General supply chain question.")
    ]
    # Compute embedding for user question
    embedding_response = openai.Embedding.create(model="text-embedding-ada-002", input=question)
    question_embedding = np.array(embedding_response['data'][0]['embedding'], dtype='float32')
    # Compute embeddings for templates
    template_embeddings = []
    for _, template in question_types:
        resp = openai.Embedding.create(model="text-embedding-ada-002", input=template)
        template_embeddings.append(np.array(resp['data'][0]['embedding'], dtype='float32'))
    # Find closest template
    sims = [np.dot(question_embedding, t) / (np.linalg.norm(question_embedding) * np.linalg.norm(t)) for t in template_embeddings]
    best_idx = int(np.argmax(sims))
    best_type = question_types[best_idx][0]

    sim_disruptions = retrieve_similar_disruptions(question_embedding)

    # --- Improved LLM prompt ---
    prompt = (
        "You are a Logistics Assistant for a supply chain company.\n"
        "You have access to the following supply chain network data: nodes (with location, operational status, risk), routes (with cost, risk, delay, status), and real-time weather.\n"
        "You can answer questions about: route status, optimal routes, operational status of locations, and general supply chain risks.\n"
        "Always answer using ONLY the provided supply chain network data. Do NOT answer with generic or internet information.\n"
        "If the question is about routes, risks, or locations, use the backend graph data to provide a detailed, data-driven answer.\n"
        "If the question is not answerable from the data, say 'I can only answer questions based on the current supply chain network.'\n"
        f"Disruptions: {sim_disruptions}\n"
        f"Question: {question}\n"
    )

    llm_text = ""
    fig = None
    # --- Weather risk summary helper ---
    def weather_risk_summary(weather):
        risks = []
        if weather['main'] in ['Rain', 'Snow', 'Thunderstorm', 'Fog', 'Drizzle']:
            risks.append(weather['main'])
        if weather['temp'] is not None and (weather['temp'] < 0 or weather['temp'] > 35):
            risks.append('Extreme temperature')
        if weather['wind'] is not None and weather['wind'] > 10:
            risks.append('High wind')
        if risks:
            return f"Weather risks: {', '.join(risks)}. Increased risk of delay."
        else:
            return "Weather risks: None detected."
    def get_weather_for_node(node_id):
        attrs = G.nodes[node_id]
        lat, lon = attrs.get('lat'), attrs.get('lon')
        if lat is None or lon is None:
            return {'main': 'Unknown', 'desc': 'Unknown', 'temp': None, 'wind': None}
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHERMAP_API_KEY}&units=metric"
        try:
            resp = requests.get(url, timeout=5)
            data = resp.json()
            main = data['weather'][0]['main'] if 'weather' in data and data['weather'] else 'Unknown'
            desc = data['weather'][0]['description'] if 'weather' in data and data['weather'] else 'Unknown'
            temp = data['main']['temp'] if 'main' in data else None
            wind = data['wind']['speed'] if 'wind' in data else None
            return {'main': main, 'desc': desc, 'temp': temp, 'wind': wind}
        except Exception as e:
            return {'main': 'Unknown', 'desc': 'Unknown', 'temp': None, 'wind': None}

    # --- Route to logic based on semantic type ---
    if best_type == "route_status":
        # Use existing route status patterns logic
        route_status_patterns = [
            r"(?:is|are) the route[s]? (?:between|from) ([\w\s,]+) (?:and|to) ([\w\s,]+) (blocked|active|inactive|open|closed)?",
            r"what is the status of the route[s]? (?:between|from) ([\w\s,]+) (?:and|to) ([\w\s,]+)",
            r"status of the route[s]? (?:between|from) ([\w\s,]+) (?:and|to) ([\w\s,]+)",
            r"route status (?:between|from) ([\w\s,]+) (?:and|to) ([\w\s,]+)"
        ]
        for pat in route_status_patterns:
            m = re.search(pat, question, re.IGNORECASE)
            if m:
                source_query = m.group(1).strip()
                target_query = m.group(2).strip()
                source = source_query if source_query in G.nodes else find_node_by_location(source_query)
                target = target_query if target_query in G.nodes else find_node_by_location(target_query)
                if not source or not target:
                    llm_text = f"Could not find nodes for '{source_query}' or '{target_query}'."
                    return llm_text
                route_infos = get_all_routes_with_status(source, target)
                if not route_infos:
                    llm_text = f"There are no routes between {source_query} and {target_query}."
                    return llm_text
                all_blocked = all(r['blocked'] for r in route_infos)
                all_active = all((not r['blocked']) for r in route_infos)
                any_active = any((not r['blocked']) for r in route_infos)
                if all_blocked:
                    llm_text = f"All routes between {G.nodes[source]['location']} and {G.nodes[target]['location']} are blocked."
                elif all_active:
                    llm_text = f"All routes between {G.nodes[source]['location']} and {G.nodes[target]['location']} are active."
                elif any_active:
                    llm_text = f"There is at least one active route between {G.nodes[source]['location']} and {G.nodes[target]['location']}, but some routes may be blocked."
                else:
                    llm_text = f"All routes between {G.nodes[source]['location']} and {G.nodes[target]['location']} are inactive or blocked."
                return llm_text
    elif best_type == "route_query":
        # Use existing route query patterns logic
        route_query_patterns = [
            r"route from ([\w\s,]+) to ([\w\s,]+)",
            r"route between ([\w\s,]+) to ([\w\s,]+)",
            r"route between ([\w\s,]+) and ([\w\s,]+)",
            r"what is the route between ([\w\s,]+) to ([\w\s,]+)",
            r"what is the route between ([\w\s,]+) and ([\w\s,]+)"
        ]
        for pat in route_query_patterns:
            m = re.search(pat, question, re.IGNORECASE)
            if m:
                source_query = m.group(1).strip()
                target_query = m.group(2).strip()
                source = source_query if source_query in G.nodes else find_node_by_location(source_query)
                target = target_query if target_query in G.nodes else find_node_by_location(target_query)
                if not source or not target:
                    llm_text = f"Could not find nodes for '{source_query}' or '{target_query}'."
                    return llm_text
                route_infos = get_all_routes_with_status(source, target)
                optimal_route = select_optimal_route(route_infos)
                fig = draw_all_routes_on_map(route_infos, optimal_route)
                # LLM description
                route_desc = describe_routes_for_llm(route_infos, optimal_route)
                # Add weather risk summary for source and target
                source_weather = get_weather_for_node(source)
                target_weather = get_weather_for_node(target)
                weather_summary = f"Source ({G.nodes[source]['location']}): {weather_risk_summary(source_weather)} | Target ({G.nodes[target]['location']}): {weather_risk_summary(target_weather)}"
                prompt += f"\n\nWeather Risk Summary: {weather_summary}\n\nRoute Analysis:\n{route_desc}"
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": question}
                    ]
                )
                llm_text = response['choices'][0]['message']['content']
                return llm_text
    elif best_type == "operational_status":
        # Use expanded operational status patterns logic
        location_status_patterns = [
            r"is the location ([\w\s,]+) active|inactive",
            r"is ([\w\s,]+) active|inactive",
            r"what is the operational status of ([\w\s,]+)",
            r"operational status of ([\w\s,]+)",
            r"what is the location status of ([\w\s,]+)",
            r"location status of ([\w\s,]+)",
            r"status of ([\w\s,]+)"
        ]
        for pat in location_status_patterns:
            m = re.search(pat, question, re.IGNORECASE)
            if m:
                loc_query = m.group(1).strip()
                node = loc_query if loc_query in G.nodes else find_node_by_location(loc_query)
                if not node:
                    llm_text = f"Could not find a location matching '{loc_query}'."
                    return llm_text
                status = G.nodes[node].get('operational_status', 'unknown').capitalize()
                llm_text = f"The operational status of {G.nodes[node]['location']} is: {status}."
                return llm_text
    # Fallback: General question, but still only use backend data
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question}
        ]
    )
    llm_text = response['choices'][0]['message']['content']
    return llm_text

def main_query_fn(user_input):
    try:
        llm_text = query_logigraph_with_agents(user_input)
        print(f"main_query_fn: llm_text={llm_text[:100]}")
        return llm_text
    except Exception as e:
        print(f"main_query_fn error: {e}")
        return f"Error: {str(e)}"

def screenshot_map(html_file, output_file, wait_time=2):
    options = Options()
    options.headless = True
    options.add_argument("--window-size=1200,800")
    driver = webdriver.Chrome(options=options)
    driver.get("file://" + os.path.abspath(html_file))
    time.sleep(wait_time)  # Wait for the map and legend to load
    driver.save_screenshot(output_file)
    driver.quit()

# --- Utility: Risk Level Distribution for Dashboard Pie Chart ---
def get_risk_level_distribution():
    risk_levels = ['low', 'medium', 'high']
    risk_counts = {level: 0 for level in risk_levels}
    for node, attrs in G.nodes(data=True):
        risk_level = attrs.get('risk_level', 'low')
        risk_counts[risk_level] += 1
    pie_labels = [level.capitalize() for level in risk_levels]
    pie_values = [risk_counts[level] for level in risk_levels]
    return pie_labels, pie_values

def get_routes_for_continent(continent):
    """Calculate the number of routes (edges) for a specific continent"""
    continent_nodes = [node for node, attrs in G.nodes(data=True) if attrs.get('continent') == continent]
    continent_edges = 0
    for u, v in G.edges():
        if u in continent_nodes and v in continent_nodes:
            continent_edges += 1
    return continent_edges

with gr.Blocks() as demo:
    gr.Markdown("## LogiGraph with Google Maps API")
    with gr.Row():
        with gr.Column():
            user_input = gr.Textbox(label="Ask a logistics question", value="What is the route from Node_1 to Node_5?")
            submit_btn = gr.Button("Submit")
            llm_response = gr.Textbox(label="LLM Response", lines=8)
        with gr.Column():
            plot_output = gr.Image(label="Route Map", height=600, width=1000)
    submit_btn.click(fn=main_query_fn, inputs=[user_input], outputs=[llm_response, plot_output])

if __name__ == "__main__":
    demo.launch(inbrowser=True)
