from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import networkx as nx
import numpy as np
import random
import faiss
import openai
from collections import defaultdict
import plotly.graph_objects as go
from geopy.geocoders import Nominatim
import pandas as pd
import os
import json
from datetime import datetime, timedelta
from SupplyChain_Backend import G, node_coords, get_lowest_risk_route, get_all_routes_with_status, select_optimal_route, get_risk_level_distribution, get_routes_for_continent
import requests
import re

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Needed for session

# --- Configuration ---
OPENAI_API_KEY = "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"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY
geolocator = Nominatim(user_agent="logigraph")
OPENWEATHERMAP_API_KEY = "a89c543f97e02a2208e1f69c8eef764a"

# --- Dashboard Data Preparation ---
def calculate_risk_score():
    high_risk = sum(1 for _, _, d in G.edges(data=True) if d['risk'] == 'high')
    medium_risk = sum(1 for _, _, d in G.edges(data=True) if d['risk'] == 'medium')
    low_risk = sum(1 for _, _, d in G.edges(data=True) if d['risk'] == 'low')
    total = high_risk + medium_risk + low_risk
    return round((high_risk * 1.0 + medium_risk * 0.5) / total * 100, 1)

def get_performance_data():
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    performance = [random.randint(85, 98) for _ in months]
    return months, performance

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username == 'admin' and password == 'admin123':
            session['logged_in'] = True
            return redirect(url_for('home'))
        else:
            error = 'Invalid username or password.'
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

@app.route('/')
def home():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    stats = {
        'total_nodes': len(G.nodes()),
        'total_edges': len(G.edges()),
        'active_nodes': sum(1 for _, d in G.nodes(data=True) if d['operational_status'] == 'active'),
        'risk_score': calculate_risk_score()
    }
    nodes_data = []
    for node, attrs in G.nodes(data=True):
        # Simulate/calculate risk factors for each node
        # Simulate delay probability (average of outgoing edges)
        delays = [edge.get('delay_probability', 0) for _, _, edge in G.edges(node, data=True)]
        delay_prob = sum(delays) / len(delays) if delays else 0
        # Operational status
        op_score = 0 if attrs.get('operational_status', 'active') == 'active' else 1
        # Fetch real weather for dashboard
        def get_weather(lat, lon):
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
        weather = get_weather(attrs.get('lat'), attrs.get('lon'))
        def weather_risk_score(w):
            risky = w['main'] in ['Rain', 'Snow', 'Thunderstorm', 'Fog', 'Drizzle']
            extreme_temp = w['temp'] is not None and (w['temp'] < 0 or w['temp'] > 35)
            high_wind = w['wind'] is not None and w['wind'] > 10
            return 1 if risky or extreme_temp or high_wind else 0
        weather_score = weather_risk_score(weather)
        # Simulate country risk
        country_risk = random.choice(['Low', 'Medium', 'High'])
        def country_risk_score(risk):
            return {'Low': 0, 'Medium': 0.5, 'High': 1}[risk]
        country_score = country_risk_score(country_risk)
        # Weighted risk
        weights = {'delay': 0.35, 'operational': 0.20, 'weather': 0.25, 'country': 0.20}
        total_risk = (
            weights['delay'] * delay_prob +
            weights['operational'] * op_score +
            weights['weather'] * weather_score +
            weights['country'] * country_score
        )
        total_risk_pct = round(total_risk * 100, 1)
        if total_risk_pct >= 66:
            qualitative = 'High'
            risk_level = 'high'
        elif total_risk_pct >= 33:
            qualitative = 'Medium'
            risk_level = 'medium'
        else:
            qualitative = 'Low'
            risk_level = 'low'
        nodes_data.append({
            'id': node,
            'location': attrs['location'],
            'lat': attrs.get('lat'),
            'lon': attrs.get('lon'),
            'continent': attrs.get('continent'),
            'status': attrs['operational_status'],
            'risk_level': risk_level,
            'risk_level_color': 'success' if risk_level == 'low' else 'warning' if risk_level == 'medium' else 'danger',
            'last_updated': attrs.get('last_updated', ''),
            'risk_factors': {
                'delay_probability': round(delay_prob * 100, 1),
                'operational_status': 'Active' if op_score == 0 else 'Inactive',
                'country_risk': country_risk,
                'weather': weather['main']
            },
            'overall_risk_qualitative': qualitative,
            'total_risk_pct': total_risk_pct
        })
    
    # Get all available locations for dropdowns
    locations = list(set([attrs['location'] for _, attrs in G.nodes(data=True)]))
    locations.sort()
    
    # Map location name to node ID
    node_location_map = {attrs['location']: node for node, attrs in G.nodes(data=True)}
    # Map node ID to location name
    node_id_to_location = {node: attrs['location'] for node, attrs in G.nodes(data=True)}
    
    # Convert all tuples to lists for JSON serialization
    serializable_node_coords = {k: list(v) for k, v in node_coords.items()}
    
    # Print country risk assignments for this session
    print('Country risk assignments for this session:')
    for node in nodes_data:
        print(f"{node['location']}: {node['risk_factors']['country_risk']}")
    
    # Calculate pie chart data from nodes_data
    risk_levels = ['low', 'medium', 'high']
    risk_counts = {level: 0 for level in risk_levels}
    for node in nodes_data:
        risk_counts[node['risk_level']] += 1
    pie_labels = [level.capitalize() for level in risk_levels]
    pie_values = [risk_counts[level] for level in risk_levels]
    return render_template(
        'index.html',
        stats=stats,
        nodes=nodes_data,
        pie_labels=pie_labels,
        pie_values=pie_values,
        locations=locations,
        node_coords=serializable_node_coords,
        node_location_map=node_location_map,
        node_id_to_location=node_id_to_location
    )

@app.route('/api/performance')
def get_performance():
    months, performance = get_performance_data()
    return jsonify({
        'months': months,
        'performance': performance
    })

@app.route('/api/risk-distribution')
def get_risk_distribution():
    high_risk = sum(1 for _, _, d in G.edges(data=True) if d['risk'] == 'high')
    medium_risk = sum(1 for _, _, d in G.edges(data=True) if d['risk'] == 'medium')
    low_risk = sum(1 for _, _, d in G.edges(data=True) if d['risk'] == 'low')
    
    return jsonify({
        'labels': ['Low Risk', 'Medium Risk', 'High Risk'],
        'values': [low_risk, medium_risk, high_risk]
    })

@app.route('/query', methods=['POST'])
def process_query():
    data = request.get_json()
    question = data.get('question', '')
    
    # Process the query using your existing functions
    llm_text, fig = query_logigraph_with_agents(question)
    
    # Convert the plotly figure to JSON
    if fig:
        fig_json = fig.to_json()
    else:
        fig_json = None
    
    return jsonify({
        'response': llm_text,
        'plot': fig_json
    })

@app.route('/route', methods=['POST'])
def get_route():
    data = request.get_json()
    source = data.get('source')
    target = data.get('target')
    
    route, risk = get_lowest_risk_route(source, target)
    if route:
        details = get_route_details(route)
        return jsonify({
            'route': route,
            'risk': risk,
            'details': details
        })
    return jsonify({'error': 'No route found'}), 404

@app.route('/risk-management')
def risk_management():
    return render_template('risk_management.html')

@app.route('/inventory-management')
def inventory_management():
    return render_template('inventory_management.html')

@app.route('/transportation-analytics')
def transportation_analytics():
    return render_template('transportation_analytics.html')

@app.route('/performance-metrics')
def performance_metrics():
    return render_template('performance_metrics.html')

@app.route('/query-interface')
def query_interface():
    return render_template('query_interface.html')

@app.route('/alerts-notifications')
def alerts_notifications():
    return render_template('alerts_notifications.html')

@app.route('/settings-configuration')
def settings_configuration():
    return render_template('settings_configuration.html')

@app.route('/api/continent-routes/<continent>')
def get_continent_routes(continent):
    """Get the number of routes for a specific continent"""
    route_count = get_routes_for_continent(continent)
    return jsonify({'routes': route_count})

@app.route('/api/chat', methods=['POST'])
def chat_with_ai():
    """Handle chat queries with OpenAI integration"""
    data = request.get_json()
    user_message = data.get('message', '')
    
    try:
        # Import the backend function for AI queries
        from SupplyChain_Backend import query_logigraph_with_agents
        
        # Get response from backend AI
        response = query_logigraph_with_agents(user_message)
        
        return jsonify({
            'success': True,
            'response': response,
            'message': user_message
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': user_message
        })

@app.route('/api/analyze-route', methods=['POST'])
def analyze_route():
    """Analyze route between two locations"""
    data = request.get_json()
    source = data.get('source')
    target = data.get('target')
    
    try:
        # Get all routes between source and target
        route_infos = get_all_routes_with_status(source, target)
        optimal_route = select_optimal_route(route_infos)
        
        # Get route details with segment breakdowns
        routes_data = []
        for info in route_infos:
            is_optimal = optimal_route and info['route'] == optimal_route['route']
            route_details = {
                'route': info['route'],
                'total_cost': info['total_cost'],
                'total_risk': info['total_risk'],
                'total_delay': info['total_delay'],
                'total_time': info['total_time'],
                'blocked': info['blocked'],
                'inactive': info['inactive'],
                'is_optimal': is_optimal,
                'segments': []
            }
            route = info['route']
            for idx in range(len(route) - 1):
                u, v = route[idx], route[idx + 1]
                edge_data = G.get_edge_data(u, v)
                if edge_data:
                    edge = list(edge_data.values())[0]
                    segment = {
                        'from': G.nodes[u]['location'],
                        'to': G.nodes[v]['location'],
                        'mode': edge.get('mode', '-'),
                        'cost': edge.get('cost_usd', '-'),
                        'risk': edge.get('risk', '-').capitalize(),
                        'delay': int(float(edge.get('delay_probability', 0)) * 100),
                        'status': edge.get('status', '-').replace('_', ' ').capitalize(),
                        'time_days': edge.get('time_days', '-')
                    }
                    route_details['segments'].append(segment)
            routes_data.append(route_details)
        
        return jsonify({
            'success': True,
            'routes': routes_data,
            'optimal_route': optimal_route['route'] if optimal_route else None,
            'source': source,
            'target': target
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/risk-management', methods=['POST'])
def risk_management_api():
    data = request.get_json()
    source = data.get('source')
    target = data.get('target')
    node_location_map = {attrs['location']: node for node, attrs in G.nodes(data=True)}
    source_id = node_location_map.get(source) if source else None
    target_id = node_location_map.get(target) if target else None
    def get_weather(lat, lon):
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
    def simulate_country_risk():
        return random.choice(['Low', 'Medium', 'High'])
    # Analyze both together (route)
    if source_id and target_id:
        source_attrs = G.nodes[source_id]
        target_attrs = G.nodes[target_id]
        source_lat, source_lon = source_attrs.get('lat'), source_attrs.get('lon')
        target_lat, target_lon = target_attrs.get('lat'), target_attrs.get('lon')
        source_weather = get_weather(source_lat, source_lon)
        target_weather = get_weather(target_lat, target_lon)
        source_country_risk = simulate_country_risk()
        target_country_risk = simulate_country_risk()
        # Get delay probability for direct edge (if exists)
        delay_prob = None
        for _, _, edge in G.edges(source_id, data=True):
            if edge.get('to') == target_id or G.nodes[edge.get('to', target_id)]['location'] == target:
                delay_prob = edge.get('delay_probability', None)
                break
        if delay_prob is None:
            delays = [edge.get('delay_probability', 0) for _, _, edge in G.edges(source_id, data=True)]
            delay_prob = sum(delays) / len(delays) if delays else 0
        # Risk scoring as before
        weights = {'delay': 0.35, 'operational': 0.20, 'weather': 0.25, 'country': 0.20}
        delay_score = delay_prob
        op_score = 0
        if source_attrs.get('operational_status', 'active') != 'active' or target_attrs.get('operational_status', 'active') != 'active':
            op_score = 1
        def weather_risk_score(w):
            risky = w['main'] in ['Rain', 'Snow', 'Thunderstorm', 'Fog', 'Drizzle']
            extreme_temp = w['temp'] is not None and (w['temp'] < 0 or w['temp'] > 35)
            high_wind = w['wind'] is not None and w['wind'] > 10
            return 1 if risky or extreme_temp or high_wind else 0
        weather_score = max(weather_risk_score(source_weather), weather_risk_score(target_weather))
        def country_risk_score(risk):
            return {'Low': 0, 'Medium': 0.5, 'High': 1}[risk]
        country_score = max(country_risk_score(source_country_risk), country_risk_score(target_country_risk))
        total_risk = (
            weights['delay'] * delay_score +
            weights['operational'] * op_score +
            weights['weather'] * weather_score +
            weights['country'] * country_score
        )
        total_risk_pct = round(total_risk * 100, 1)
        if total_risk_pct >= 66:
            qualitative = 'High'
        elif total_risk_pct >= 33:
            qualitative = 'Medium'
        else:
            qualitative = 'Low'
        def weather_risk(w):
            risks = []
            if w['main'] in ['Rain', 'Snow', 'Thunderstorm', 'Fog', 'Drizzle']:
                risks.append(w['main'])
            if w['temp'] is not None and (w['temp'] < 0 or w['temp'] > 35):
                risks.append('Extreme temperature')
            if w['wind'] is not None and w['wind'] > 10:
                risks.append('High wind')
            return ', '.join(risks) if risks else 'None'
        summary = f"Risk analysis from {source} to {target}:\n"
        summary += f"- Source status: {source_attrs.get('operational_status', 'unknown')}, Weather: {source_weather['desc']} (Risks: {weather_risk(source_weather)}), Country risk: {source_country_risk}\n"
        summary += f"- Target status: {target_attrs.get('operational_status', 'unknown')}, Weather: {target_weather['desc']} (Risks: {weather_risk(target_weather)}), Country risk: {target_country_risk}\n"
        summary += f"- Estimated delay probability: {round(delay_prob*100,1)}% (Weight: 35%)\n"
        summary += f"- Operational status risk: {'High' if op_score == 1 else 'Low'} (Weight: 20%)\n"
        summary += f"- Weather risk: {'High' if weather_score == 1 else 'Low'} (Weight: 25%)\n"
        summary += f"- Country risk: {'High' if country_score == 1 else 'Medium' if country_score == 0.5 else 'Low'} (Weight: 20%)\n"
        summary += f"- Total route risk: {total_risk_pct}% ({qualitative})\n"
        summary += f"\nBreakdown of risk calculation:\n"
        summary += f"  - Delay risk: {round(delay_score*100,1)}% x 0.35 = {round(delay_score*35,1)}\n"
        summary += f"  - Operational status: {op_score} x 0.20 = {round(op_score*20,1)}\n"
        summary += f"  - Weather risk: {weather_score} x 0.25 = {round(weather_score*25,1)}\n"
        summary += f"  - Country risk: {country_score} x 0.20 = {round(country_score*20,1)}\n"
        summary += f"  - Total weighted risk: {total_risk_pct}%\n"
        if qualitative == 'High':
            summary += "\nOverall risk: HIGH. Consider alternate routes or mitigation strategies."
        elif qualitative == 'Medium':
            summary += "\nOverall risk: MEDIUM. Monitor closely and prepare contingencies."
        else:
            summary += "\nOverall risk: LOW. Route is generally safe."
        return jsonify({'success': True, 'result': {
            'source': source,
            'target': target,
            'source_status': source_attrs.get('operational_status', 'unknown'),
            'target_status': target_attrs.get('operational_status', 'unknown'),
            'source_weather': source_weather,
            'target_weather': target_weather,
            'source_country_risk': source_country_risk,
            'target_country_risk': target_country_risk,
            'delay_probability': round(delay_prob * 100, 1),
            'total_risk_pct': total_risk_pct,
            'total_risk_qualitative': qualitative,
            'summary': summary
        }})
    return jsonify({'success': False, 'error': 'Please select at least one valid location.'})

@app.route('/api/inventory-details', methods=['POST'])
def inventory_details_api():
    data = request.get_json()
    location = data.get('location')
    node_location_map = {attrs['location']: node for node, attrs in G.nodes(data=True)}
    node_id = node_location_map.get(location)
    if not node_id:
        return jsonify({'success': False, 'error': 'Invalid location.'})
    attrs = G.nodes[node_id]
    lat, lon = attrs.get('lat'), attrs.get('lon')
    def get_weather(lat, lon):
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
    weather = get_weather(lat, lon)
    # Load agent info from CSV
    agent_info = None
    try:
        df_agents = pd.read_csv('agent_database.csv')
        # Strip whitespace from all string columns
        df_agents = df_agents.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        agent_row = df_agents[df_agents['Location'].str.lower() == location.strip().lower()]
        if not agent_row.empty:
            row = agent_row.iloc[0]
            agent_info = {
                'name': row['Name'],
                'role': row['Role'],
                'location': row['Location'],
                'status': row['Status'],
                'phone': row['Contact Number'],
                'email': row['Email']
            }
        else:
            agent_info = {
                'name': f"Agent for {location}",
                'role': '-',
                'location': location,
                'status': '-',
                'phone': '-',
                'email': '-'
            }
    except Exception as e:
        agent_info = {
            'name': f"Agent for {location}",
            'role': '-',
            'location': location,
            'status': '-',
            'phone': '-',
            'email': '-'
        }
    hours = random.choice([
        '24/7',
        'Mon-Fri 8am-8pm',
        'Mon-Sat 6am-10pm',
        'Mon-Fri 7am-7pm, Sat 8am-4pm'
    ])
    result = {
        'location': location,
        'storage_capacity': attrs.get('storage_capacity_tons', 'N/A'),
        'available_space': attrs.get('inventory_level', 'N/A'),
        'hours': hours,
        'agent': agent_info,
        'weather': weather
    }
    return jsonify({'success': True, 'result': result})

@app.route('/api/ai-risk-expert', methods=['POST'])
def ai_risk_expert():
    data = request.get_json()
    query = data.get('query', '').strip()
    if not query:
        return jsonify({'success': False, 'error': 'No query provided.'})
    # Load agent database
    try:
        df_agents = pd.read_csv('agent_database.csv')
        df_agents = df_agents.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        agent_info = df_agents.to_dict(orient='records')
    except Exception as e:
        agent_info = []
    # Gather graph data
    from SupplyChain_Backend import G
    nodes_data = []
    for node, attrs in G.nodes(data=True):
        nodes_data.append({**attrs, 'id': node})
    edges_data = []
    for u, v, attrs in G.edges(data=True):
        edges_data.append({'source': u, 'target': v, **attrs})
    # Try to extract source and target locations from the query
    source, target = None, None
    match = re.search(r'from ([\w\s,]+) to ([\w\s,]+)', query, re.IGNORECASE)
    if match:
        source = match.group(1).strip()
        target = match.group(2).strip()
    # Try to extract a single location if not a route query
    single_location = None
    country_locations = []
    if not source and not target:
        # Try to find a location in the query that matches any node location
        for node in nodes_data:
            loc = node['location']
            if loc.lower() in query.lower():
                single_location = loc
                break
        # Country-level matching: if no city match, try to match by country
        if not single_location:
            # Extract possible country from query (last word or after 'in')
            words = query.lower().split()
            possible_country = words[-1]
            # Also check for 'in [country]' pattern
            match_country = re.search(r'in ([\w\s]+)', query, re.IGNORECASE)
            if match_country:
                possible_country = match_country.group(1).strip().lower()
            # Find all nodes whose location ends with the country
            for node in nodes_data:
                loc = node['location']
                if loc.lower().endswith(possible_country):
                    country_locations.append(loc)
    # Get weather for source, target, single location, or country locations if found
    weather_info = {}
    def get_weather_for_location(location):
        node = next((n for n in nodes_data if n['location'].lower() == location.lower()), None)
        if node and node.get('lat') is not None and node.get('lon') is not None:
            lat, lon = node['lat'], node['lon']
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
        return {'main': 'Unknown', 'desc': 'Unknown', 'temp': None, 'wind': None}
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
    # Get weather for source, target, single location, or country locations if found
    weather_summaries = []
    if source:
        w = get_weather_for_location(source)
        weather_info['source'] = {'location': source, 'weather': w}
        weather_summaries.append(f"Source ({source}): {weather_risk_summary(w)}")
    if target:
        w = get_weather_for_location(target)
        weather_info['target'] = {'location': target, 'weather': w}
        weather_summaries.append(f"Target ({target}): {weather_risk_summary(w)}")
    if single_location:
        w = get_weather_for_location(single_location)
        weather_info['location'] = {'location': single_location, 'weather': w}
        weather_summaries.append(f"Location ({single_location}): {weather_risk_summary(w)}")
    if country_locations:
        weather_info['country'] = []
        for loc in country_locations:
            w = get_weather_for_location(loc)
            weather_info['country'].append({'location': loc, 'weather': w})
            weather_summaries.append(f"Location ({loc}): {weather_risk_summary(w)}")
    # Build context string
    context = f"Agents: {agent_info}\nNodes: {nodes_data}\nRoutes: {edges_data}\nWeather: {weather_info}\nWeather Risk Summary: {' | '.join(weather_summaries)}"
    prompt = (
        "You are an AI Risk Expert for a supply chain system. "
        "You have access to the following data: agent database, locations, routes, weather, delay probability, country risk, and route status. "
        "Always answer using ONLY the provided supply chain data and the real-time weather data. "
        "If the question is about risks, locations, routes, weather, delay probability, or country risk, use the data to provide a precise, actionable recommendation. "
        "If the question cannot be answered from the data, say 'I can only answer questions based on the current supply chain data.'\n"
        f"Supply Chain Data:\n{context}\n\nUser Query: {query}"
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful AI Risk Expert. Only use the provided supply chain data and real-time weather for your answers."},
                {"role": "user", "content": prompt}
            ]
        )
        answer = response['choices'][0]['message']['content']
        return jsonify({'success': True, 'response': answer})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000) 