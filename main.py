import os
import heapq

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from databricks import sql


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

app = FastAPI(
    title="MetroFlow API",
    description="Smart Crowd-Aware Delhi Metro Journey Planner",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# DATABRICKS CONNECTION
# ============================================================

def get_connection():
    return sql.connect(
        server_hostname=os.getenv("DATABRICKS_SERVER_HOSTNAME"),
        http_path=os.getenv("DATABRICKS_HTTP_PATH"),
        access_token=os.getenv("DATABRICKS_TOKEN")
    )


# ============================================================
# NETWORK CACHE
# ============================================================

metro_graph = None
station_names = None
station_lookup = None


def load_network():

    global metro_graph
    global station_names
    global station_lookup

    if metro_graph is not None:
        return

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            from_station_id,
            from_station_name,
            to_station_id,
            to_station_name,
            metro_line,
            travel_time_minutes
        FROM workspace.metroflow_gold.gold_route_serving
    """)

    rows = cursor.fetchall()

    cursor.close()
    conn.close()

    metro_graph = {}
    station_names = {}

    for row in rows:

        from_id = str(row[0])
        from_name = row[1]

        to_id = str(row[2])
        to_name = row[3]

        metro_line = row[4]
        travel_time = float(row[5])

        station_names[from_id] = from_name
        station_names[to_id] = to_name

        if from_id not in metro_graph:
            metro_graph[from_id] = []

        metro_graph[from_id].append({
            "station_id": to_id,
            "station_name": to_name,
            "metro_line": metro_line,
            "travel_time": travel_time
        })

    station_lookup = {
        name.strip().lower(): station_id
        for station_id, name in station_names.items()
    }


# ============================================================
# DIJKSTRA ROUTE FINDER
# ============================================================

def calculate_route(
    source,
    destination,
    blocked_edges=None
):

    load_network()

    if blocked_edges is None:
        blocked_edges = set()

    distances = {
        station_id: float("inf")
        for station_id in metro_graph
    }

    previous = {
        station_id: None
        for station_id in metro_graph
    }

    distances[source] = 0

    priority_queue = [
        (0, source)
    ]

    while priority_queue:

        current_distance, current_station = heapq.heappop(
            priority_queue
        )

        if current_distance > distances[current_station]:
            continue

        if current_station == destination:
            break

        for edge in metro_graph.get(current_station, []):

            next_station = edge["station_id"]
            metro_line = edge["metro_line"]

            edge_key = (
                current_station,
                next_station,
                metro_line
            )

            if edge_key in blocked_edges:
                continue

            new_distance = (
                current_distance
                + edge["travel_time"]
            )

            if new_distance < distances[next_station]:

                distances[next_station] = new_distance

                previous[next_station] = {
                    "station_id": current_station,
                    "metro_line": metro_line
                }

                heapq.heappush(
                    priority_queue,
                    (
                        new_distance,
                        next_station
                    )
                )

    # No route
    if distances[destination] == float("inf"):
        return None

    # --------------------------------------------------------
    # Reconstruct station path
    # --------------------------------------------------------

    route_ids = []

    current = destination

    while current is not None:

        route_ids.append(current)

        if previous[current] is None:
            break

        current = previous[current]["station_id"]

    route_ids.reverse()

    # --------------------------------------------------------
    # Build station-by-station line information
    # --------------------------------------------------------

    station_details = []

    current_line = None

    for i, station_id in enumerate(route_ids):

        station_name = station_names[station_id]

        # ----------------------------------------------------
        # Last station
        # ----------------------------------------------------

        if i == len(route_ids) - 1:

            station_details.append({
                "station_id": station_id,
                "station_name": station_name,
                "line": current_line,
                "is_interchange": False,
                "from_line": current_line,
                "to_line": None
            })

            break

        next_station = route_ids[i + 1]

        previous_info = previous[next_station]

        line = previous_info["metro_line"]

        # ----------------------------------------------------
        # Detect interchange
        # ----------------------------------------------------

        is_interchange = (
            current_line is not None
            and line != current_line
        )

        station_details.append({
            "station_id": station_id,
            "station_name": station_name,
            "line": line,
            "is_interchange": is_interchange,
            "from_line": current_line,
            "to_line": line if is_interchange else None
        })

        current_line = line

    # --------------------------------------------------------
    # Unique metro lines
    # --------------------------------------------------------

    lines = []

    for station in station_details:

        line = station["line"]

        if line and line not in lines:
            lines.append(line)

    # --------------------------------------------------------
    # Interchange count
    # --------------------------------------------------------

    interchange_count = sum(
        1
        for station in station_details
        if station["is_interchange"]
    )

    # --------------------------------------------------------
    # Edge line sequence
    # --------------------------------------------------------

    edge_lines = [
        station["line"]
        for station in station_details[:-1]
    ]

    return {
        "station_ids": route_ids,

        "stations": [
            station_names[x]
            for x in route_ids
        ],

        "station_details": station_details,

        "travel_time_minutes": round(
            distances[destination],
            2
        ),

        "lines_used": lines,

        "interchange_count": interchange_count,

        "edge_lines": edge_lines
    }


# ============================================================
# FIND MULTIPLE ROUTES
# ============================================================

def find_multiple_routes(
    source_name,
    destination_name
):

    load_network()

    source = station_lookup.get(
        source_name.strip().lower()
    )

    destination = station_lookup.get(
        destination_name.strip().lower()
    )

    if source is None:

        raise HTTPException(
            status_code=404,
            detail=f"Source station not found: {source_name}"
        )

    if destination is None:

        raise HTTPException(
            status_code=404,
            detail=f"Destination station not found: {destination_name}"
        )

    # --------------------------------------------------------
    # Main shortest route
    # --------------------------------------------------------

    main_route = calculate_route(
        source,
        destination
    )

    if main_route is None:

        raise HTTPException(
            status_code=404,
            detail="No route found"
        )

    candidates = [
        main_route
    ]

    # --------------------------------------------------------
    # Generate alternatives
    # --------------------------------------------------------

    main_ids = main_route["station_ids"]
    main_lines = main_route["edge_lines"]

    for i in range(len(main_ids) - 1):

        blocked_edge = {
            (
                main_ids[i],
                main_ids[i + 1],
                main_lines[i]
            )
        }

        alternative = calculate_route(
            source,
            destination,
            blocked_edges=blocked_edge
        )

        if alternative is None:
            continue

        existing_routes = {
            tuple(route["station_ids"])
            for route in candidates
        }

        alternative_key = tuple(
            alternative["station_ids"]
        )

        if alternative_key not in existing_routes:

            candidates.append(
                alternative
            )

        if len(candidates) >= 3:
            break

    return candidates[:3]


# ============================================================
# CROWD DATA
# ============================================================

def get_crowd_for_route(
    station_ids,
    departure_hour
):

    if not station_ids:
        return []

    conn = get_connection()
    cursor = conn.cursor()

    placeholders = ",".join(
        ["?"] * len(station_ids)
    )

    query = f"""
        SELECT
            station_id,
            station_name,
            hour,
            scheduled_train_visits,
            activity_level,
            crowd_score,
            crowd_level,
            seat_likelihood_pct
        FROM workspace.metroflow_gold.gold_crowd_serving
        WHERE hour = ?
        AND station_id IN ({placeholders})
    """

    params = [
        departure_hour
    ] + station_ids

    cursor.execute(
        query,
        params
    )

    rows = cursor.fetchall()

    cursor.close()
    conn.close()

    result = []

    for row in rows:

        result.append({
            "station_id": str(row[0]),
            "station_name": row[1],
            "hour": row[2],
            "scheduled_train_visits": row[3],
            "activity_level": row[4],
            "crowd_score": float(row[5]),
            "crowd_level": row[6],
            "estimated_seat_likelihood_pct": float(row[7])
        })

    return result


# ============================================================
# CROWD SUMMARY
# ============================================================

def calculate_crowd_summary(crowd_data):

    if not crowd_data:
        return {
            "average_crowd_score": None,
            "average_estimated_seat_likelihood_pct": None,
            "worst_estimated_seat_likelihood_pct": None,
            "worst_station_name": None,
            "maximum_crowd_score": None
        }

    crowd_scores = [
        x["crowd_score"]
        for x in crowd_data
    ]

    seat_values = [
        x["estimated_seat_likelihood_pct"]
        for x in crowd_data
    ]

    # Station with lowest estimated seat likelihood
    worst_station = min(
        crowd_data,
        key=lambda x: x["estimated_seat_likelihood_pct"]
    )

    return {
        "average_crowd_score":
            round(
                sum(crowd_scores) / len(crowd_scores),
                2
            ),

        "average_estimated_seat_likelihood_pct":
            round(
                sum(seat_values) / len(seat_values),
                2
            ),

        "worst_estimated_seat_likelihood_pct":
            round(
                worst_station[
                    "estimated_seat_likelihood_pct"
                ],
                2
            ),

        "worst_station_name":
            worst_station["station_name"],

        "maximum_crowd_score":
            round(
                max(crowd_scores),
                2
            )
    }


# ============================================================
# ROUTE SCORE
# ============================================================

def calculate_route_score(
    route
):

    crowd = route["crowd_summary"]

    travel_time = route[
        "travel_time_minutes"
    ]

    average_crowd = crowd[
        "average_crowd_score"
    ]

    average_seat = crowd[
        "average_estimated_seat_likelihood_pct"
    ]

    interchanges = route[
        "interchange_count"
    ]

    if average_crowd is None:
        return 0

    time_score = max(
        0,
        100 - (travel_time * 2)
    )

    crowd_score = (
        100 - average_crowd
    )

    seat_score = average_seat

    interchange_score = max(
        0,
        100 - (interchanges * 25)
    )

    score = (
        (time_score * 0.30)
        + (crowd_score * 0.25)
        + (seat_score * 0.30)
        + (interchange_score * 0.15)
    )

    return round(
        score,
        2
    )


# ============================================================
# TIME PARSER
# ============================================================

def parse_departure_time(
    departure_time
):

    try:

        hour, minute = departure_time.split(":")

        hour = int(hour)
        minute = int(minute)

    except ValueError:

        raise HTTPException(
            status_code=400,
            detail="Departure time must be HH:MM"
        )

    if hour < 0 or hour > 23:

        raise HTTPException(
            status_code=400,
            detail="Hour must be between 00 and 23"
        )

    if minute < 0 or minute > 59:

        raise HTTPException(
            status_code=400,
            detail="Minute must be between 00 and 59"
        )

    return hour, minute


# ============================================================
# LABEL ROUTES
# ============================================================

def label_routes(routes):

    if not routes:
        return routes

    # --------------------------------------------------------
    # FASTEST
    # --------------------------------------------------------

    fastest = min(
        routes,
        key=lambda x:
        x["travel_time_minutes"]
    )

    # --------------------------------------------------------
    # COMFORTABLE
    # Lowest crowd
    # --------------------------------------------------------

    comfortable = min(
        routes,
        key=lambda x:
        x["crowd_summary"]["average_crowd_score"]
        if x["crowd_summary"]["average_crowd_score"]
        is not None
        else 999
    )

    # --------------------------------------------------------
    # BALANCED
    # Highest trade-off score
    # --------------------------------------------------------

    balanced = max(
        routes,
        key=lambda x:
        x["route_score"]
    )

    # --------------------------------------------------------
    # Reset labels
    # --------------------------------------------------------

    for route in routes:
        route["type"] = "ALTERNATIVE"

    fastest["type"] = "FASTEST"
    comfortable["type"] = "COMFORTABLE"
    balanced["type"] = "BALANCED"

    return routes


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {
        "message": "MetroFlow API is running",
        "version": "1.0.0"
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }

@app.get("/stations")
def get_stations():
    load_network()

    return {
        "stations": sorted(
            station_names.values(),
            key=lambda x: x.lower()
        )
    }




# ============================================================
# SINGLE ROUTE
# ============================================================

@app.get("/route")
def route(
    source: str,
    destination: str
):

    routes = find_multiple_routes(
        source,
        destination
    )

    result = routes[0]

    return {
        "source": source,
        "destination": destination,
        "travel_time_minutes":
            result["travel_time_minutes"],
        "interchange_count":
            result["interchange_count"],
        "lines_used":
            result["lines_used"],
        "stations":
            result["stations"],
        "station_details":
            result["station_details"]
    }


# ============================================================
# SINGLE STATION CROWD
# ============================================================

@app.get("/crowd")
def crowd(
    station: str,
    hour: int
):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            station_name,
            hour,
            scheduled_train_visits,
            activity_level,
            crowd_score,
            crowd_level,
            seat_likelihood_pct
        FROM workspace.metroflow_gold.gold_crowd_serving
        WHERE LOWER(station_name) = LOWER(?)
        AND hour = ?
    """, (
        station,
        hour
    ))

    row = cursor.fetchone()

    cursor.close()
    conn.close()

    if not row:

        raise HTTPException(
            status_code=404,
            detail="Crowd data not found"
        )

    return {
        "station": row[0],
        "hour": row[1],
        "scheduled_train_visits": row[2],
        "activity_level": row[3],
        "crowd_score": row[4],
        "crowd_level": row[5],
        "estimated_seat_likelihood_pct":
            row[6]
    }


# ============================================================
# COMPLETE JOURNEY PLANNER
# ============================================================

@app.get("/plan")
def plan(
    source: str,
    destination: str,
    departure_time: str
):

    # --------------------------------------------------------
    # Parse time
    # --------------------------------------------------------

    departure_hour, departure_minute = (
        parse_departure_time(
            departure_time
        )
    )

    # --------------------------------------------------------
    # Find routes
    # --------------------------------------------------------

    raw_routes = find_multiple_routes(
        source,
        destination
    )

    final_routes = []

    # --------------------------------------------------------
    # Crowd analysis
    # --------------------------------------------------------

    for route in raw_routes:

        crowd_data = get_crowd_for_route(
            route["station_ids"],
            departure_hour
        )

        crowd_summary = calculate_crowd_summary(
            crowd_data
        )

        route_result = {

            "travel_time_minutes":
                route["travel_time_minutes"],

            "interchange_count":
                route["interchange_count"],

            "lines_used":
                route["lines_used"],

            "stations":
                route["stations"],

            "station_details":
                route["station_details"],

            "crowd_summary":
                crowd_summary,

            "station_crowd":
                crowd_data
        }

        final_routes.append(
            route_result
        )

    # --------------------------------------------------------
    # Calculate trade-off score
    # --------------------------------------------------------

    for route in final_routes:

        route["route_score"] = (
            calculate_route_score(
                route
            )
        )

    # --------------------------------------------------------
    # Label routes
    # --------------------------------------------------------

    final_routes = label_routes(
        final_routes
    )

    # --------------------------------------------------------
    # Sort routes
    # --------------------------------------------------------

    type_order = {
        "FASTEST": 1,
        "COMFORTABLE": 2,
        "BALANCED": 3,
        "ALTERNATIVE": 4
    }

    final_routes.sort(
        key=lambda x:
        type_order.get(
            x["type"],
            99
        )
    )

    # --------------------------------------------------------
    # Final response
    # --------------------------------------------------------

    return {

        "source":
            source,

        "destination":
            destination,

        "departure_time":
            departure_time,

        "crowd_profile_hour":
            departure_hour,

        "routes":
            final_routes,

        "route_count":
            len(final_routes),

        "disclaimer":
            "Crowd and seat values are modelled estimates "
            "based on scheduled service activity and are "
            "not real-time occupancy or guaranteed seat "
            "availability."
    }