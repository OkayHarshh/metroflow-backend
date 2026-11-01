import os
import csv
import heapq
import time
import threading

from threading import Lock
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
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# DATABRICKS CONFIG
# ============================================================

DATABRICKS_RETRIES = 3
DATABRICKS_RETRY_DELAY = 2


def get_connection():
    hostname = os.getenv("DATABRICKS_SERVER_HOSTNAME")
    http_path = os.getenv("DATABRICKS_HTTP_PATH")
    token = os.getenv("DATABRICKS_TOKEN")

    if not hostname or not http_path or not token:
        raise RuntimeError(
            "Databricks environment variables are missing"
        )

    return sql.connect(
        server_hostname=hostname,
        http_path=http_path,
        access_token=token
    )


def execute_databricks_query(query):
    """
    Executes a Databricks query with retries.
    Temporary Databricks failures won't immediately
    break the user request.
    """

    last_error = None

    for attempt in range(1, DATABRICKS_RETRIES + 1):

        conn = None
        cursor = None

        try:
            conn = get_connection()
            cursor = conn.cursor()

            cursor.execute(query)
            rows = cursor.fetchall()

            return rows

        except Exception as e:
            last_error = e

            print(
                f"Databricks attempt "
                f"{attempt}/{DATABRICKS_RETRIES} failed: {e}"
            )

            if attempt < DATABRICKS_RETRIES:
                time.sleep(
                    DATABRICKS_RETRY_DELAY * attempt
                )

        finally:
            if cursor:
                try:
                    cursor.close()
                except Exception:
                    pass

            if conn:
                try:
                    conn.close()
                except Exception:
                    pass

    raise RuntimeError(
        f"Databricks request failed after "
        f"{DATABRICKS_RETRIES} attempts: {last_error}"
    )


# ============================================================
# LOCAL STATION DATA
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

STATIONS_FILE = os.path.join(
    BASE_DIR,
    "dim_station.csv"
)

local_stations = []
local_station_lookup = {}
_local_stations_lock = Lock()


def load_local_stations():

    global local_stations
    global local_station_lookup

    if local_stations:
        return

    with _local_stations_lock:

        # Double-check after acquiring lock
        if local_stations:
            return

        try:

            stations = []
            lookup = {}

            with open(
                STATIONS_FILE,
                mode="r",
                encoding="utf-8-sig"
            ) as file:

                reader = csv.DictReader(file)

                for row in reader:

                    station_name = (
                        row.get("station_name") or ""
                    ).strip()

                    station_id = (
                        row.get("station_id") or ""
                    ).strip()

                    if station_name:

                        stations.append(
                            station_name
                        )

                        lookup[
                            station_name.lower()
                        ] = station_id

            # Remove duplicates and sort
            local_stations = sorted(
                list(set(stations)),
                key=lambda x: x.lower()
            )

            local_station_lookup = lookup

            print(
                f"Loaded "
                f"{len(local_stations)} stations "
                f"from local CSV"
            )

        except FileNotFoundError:

            raise RuntimeError(
                f"Station file not found: "
                f"{STATIONS_FILE}"
            )


# ============================================================
# NETWORK CACHE
# ============================================================

metro_graph = None
station_names = None
station_lookup = None

_network_lock = Lock()


def load_network():

    global metro_graph
    global station_names
    global station_lookup

    # Already available in memory
    if metro_graph is not None:
        return

    with _network_lock:

        # Double-check after acquiring lock
        if metro_graph is not None:
            return

        print(
            "Loading route network from Databricks..."
        )

        query = """
            SELECT
                from_station_id,
                from_station_name,
                to_station_id,
                to_station_name,
                metro_line,
                travel_time_minutes
            FROM workspace.metroflow_gold.gold_route_serving
        """

        rows = execute_databricks_query(query)

        graph = {}
        names = {}

        for row in rows:

            from_id = str(row[0])
            from_name = row[1]

            to_id = str(row[2])
            to_name = row[3]

            metro_line = row[4]
            travel_time = float(row[5])

            names[from_id] = from_name
            names[to_id] = to_name

            if from_id not in graph:
                graph[from_id] = []

            graph[from_id].append({
                "station_id": to_id,
                "station_name": to_name,
                "metro_line": metro_line,
                "travel_time": travel_time
            })

        # Make sure destination-only stations exist
        for station_id in names:
            if station_id not in graph:
                graph[station_id] = []

        lookup = {
            name.strip().lower(): station_id
            for station_id, name in names.items()
            if name
        }

        # Assign only after successful complete load
        metro_graph = graph
        station_names = names
        station_lookup = lookup

        print(
            f"Route network loaded successfully: "
            f"{len(station_names)} stations"
        )


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

    if source not in metro_graph:
        return None

    if destination not in metro_graph:
        return None

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

        current_distance, current_station = (
            heapq.heappop(priority_queue)
        )

        if (
            current_distance
            > distances[current_station]
        ):
            continue

        if current_station == destination:
            break

        for edge in metro_graph.get(
            current_station,
            []
        ):

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

            if (
                new_distance
                < distances[next_station]
            ):

                distances[next_station] = (
                    new_distance
                )

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

    # No route found
    if distances[destination] == float("inf"):
        return None

    # --------------------------------------------------------
    # Reconstruct path
    # --------------------------------------------------------

    route_ids = []

    current = destination

    while current is not None:

        route_ids.append(current)

        if previous[current] is None:
            break

        current = previous[current][
            "station_id"
        ]

    route_ids.reverse()

    # --------------------------------------------------------
    # Build station details
    # --------------------------------------------------------

    station_details = []
    current_line = None

    for i, station_id in enumerate(route_ids):

        station_name = station_names[
            station_id
        ]

        # Last station
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

        line = previous[next_station][
            "metro_line"
        ]

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
            "to_line": (
                line
                if is_interchange
                else None
            )
        })

        current_line = line

    # --------------------------------------------------------
    # Unique lines
    # --------------------------------------------------------

    lines = []

    for station in station_details:

        line = station["line"]

        if line and line not in lines:
            lines.append(line)

    interchange_count = sum(
        1
        for station in station_details
        if station["is_interchange"]
    )

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
            detail=(
                f"Source station not found: "
                f"{source_name}"
            )
        )

    if destination is None:

        raise HTTPException(
            status_code=404,
            detail=(
                f"Destination station not found: "
                f"{destination_name}"
            )
        )

    if source == destination:

        raise HTTPException(
            status_code=400,
            detail=(
                "Source and destination "
                "cannot be the same"
            )
        )

    main_route = calculate_route(
        source,
        destination
    )

    if main_route is None:

        raise HTTPException(
            status_code=404,
            detail="No route found"
        )

    candidates = [main_route]

    main_ids = main_route["station_ids"]
    main_lines = main_route["edge_lines"]

    # Generate alternative routes
    for i in range(
        min(
            len(main_ids) - 1,
            len(main_lines)
        )
    ):

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
            tuple(
                route["station_ids"]
            )
            for route in candidates
        }

        alternative_key = tuple(
            alternative["station_ids"]
        )

        if (
            alternative_key
            not in existing_routes
        ):
            candidates.append(alternative)

        if len(candidates) >= 3:
            break

    return candidates[:3]


# ============================================================
# CROWD DATA CACHE
# ============================================================

CROWD_CACHE_TTL_SECONDS = 300

_crowd_cache = None
_crowd_cache_loaded_at = 0.0

_crowd_cache_lock = Lock()


def load_crowd_cache(
    force_refresh=False
):

    global _crowd_cache
    global _crowd_cache_loaded_at

    now = time.monotonic()

    cache_is_fresh = (
        _crowd_cache is not None
        and (
            now - _crowd_cache_loaded_at
            < CROWD_CACHE_TTL_SECONDS
        )
    )

    if (
        cache_is_fresh
        and not force_refresh
    ):
        return _crowd_cache

    with _crowd_cache_lock:

        now = time.monotonic()

        cache_is_fresh = (
            _crowd_cache is not None
            and (
                now - _crowd_cache_loaded_at
                < CROWD_CACHE_TTL_SECONDS
            )
        )

        if (
            cache_is_fresh
            and not force_refresh
        ):
            return _crowd_cache

        print(
            "Loading crowd data from Databricks..."
        )

        query = """
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
        """

        rows = execute_databricks_query(query)

        cache = {}

        for row in rows:

            station_id = str(row[0])
            hour = int(row[2])

            cache[
                (station_id, hour)
            ] = {
                "station_id": station_id,
                "station_name": row[1],
                "hour": hour,
                "scheduled_train_visits": row[3],
                "activity_level": row[4],
                "crowd_score": float(row[5]),
                "crowd_level": row[6],
                "estimated_seat_likelihood_pct":
                    float(row[7])
            }

        # Only replace cache after successful load
        _crowd_cache = cache

        _crowd_cache_loaded_at = (
            time.monotonic()
        )

        print(
            f"Crowd cache loaded: "
            f"{len(cache)} records"
        )

        return _crowd_cache


def get_crowd_for_route(
    station_ids,
    departure_hour
):

    if not station_ids:
        return []

    cache = load_crowd_cache()

    result = []
    seen = set()

    for station_id in station_ids:

        key = (
            str(station_id),
            int(departure_hour)
        )

        if key in cache and key not in seen:

            result.append(
                cache[key]
            )

            seen.add(key)

    return result


# ============================================================
# CROWD SUMMARY
# ============================================================

def calculate_crowd_summary(crowd_data):

    if not crowd_data:

        return {
            "average_crowd_score": None,
            "average_estimated_seat_likelihood_pct":
                None,
            "worst_estimated_seat_likelihood_pct":
                None,
            "worst_station_name": None,
            "maximum_crowd_score": None
        }

    crowd_scores = [
        x["crowd_score"]
        for x in crowd_data
    ]

    seat_values = [
        x[
            "estimated_seat_likelihood_pct"
        ]
        for x in crowd_data
    ]

    worst_station = min(
        crowd_data,
        key=lambda x:
            x[
                "estimated_seat_likelihood_pct"
            ]
    )

    return {
        "average_crowd_score": round(
            sum(crowd_scores)
            / len(crowd_scores),
            2
        ),

        "average_estimated_seat_likelihood_pct":
            round(
                sum(seat_values)
                / len(seat_values),
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
            worst_station[
                "station_name"
            ],

        "maximum_crowd_score":
            round(
                max(crowd_scores),
                2
            )
    }


# ============================================================
# ROUTE SCORE
# ============================================================

def calculate_route_score(route):

    crowd = route["crowd_summary"]

    average_crowd = crowd[
        "average_crowd_score"
    ]

    average_seat = crowd[
        "average_estimated_seat_likelihood_pct"
    ]

    if average_crowd is None:
        return 0

    travel_time = route[
        "travel_time_minutes"
    ]

    interchanges = route[
        "interchange_count"
    ]

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

    return round(score, 2)


# ============================================================
# TIME PARSER
# ============================================================

def parse_departure_time(
    departure_time
):

    try:

        hour, minute = (
            departure_time.split(":")
        )

        hour = int(hour)
        minute = int(minute)

    except (
        ValueError,
        AttributeError
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Departure time must "
                "be in HH:MM format"
            )
        )

    if hour < 0 or hour > 23:

        raise HTTPException(
            status_code=400,
            detail=(
                "Hour must be "
                "between 00 and 23"
            )
        )

    if minute < 0 or minute > 59:

        raise HTTPException(
            status_code=400,
            detail=(
                "Minute must be "
                "between 00 and 59"
            )
        )

    return hour, minute


# ============================================================
# LABEL ROUTES
# ============================================================

def label_routes(routes):

    if not routes:
        return routes

    fastest = min(
        routes,
        key=lambda x:
            x["travel_time_minutes"]
    )

    comfortable = min(
        routes,
        key=lambda x:
            x["crowd_summary"][
                "average_crowd_score"
            ]
            if x["crowd_summary"][
                "average_crowd_score"
            ] is not None
            else float("inf")
    )

    balanced = max(
        routes,
        key=lambda x:
            x["route_score"]
    )

    for route in routes:
        route["type"] = "ALTERNATIVE"

    fastest["type"] = "FASTEST"
    comfortable["type"] = "COMFORTABLE"

    # Don't overwrite FASTEST/COMFORTABLE
    # unnecessarily when routes overlap.
    if balanced["type"] == "ALTERNATIVE":
        balanced["type"] = "BALANCED"

    return routes


# ============================================================
# BACKGROUND WARM-UP
# ============================================================

def warm_up_databricks():

    try:

        print(
            "Warming up route network..."
        )

        load_network()

        print(
            "Route network ready"
        )

        print(
            "Warming up crowd cache..."
        )

        load_crowd_cache()

        print(
            "Crowd cache ready"
        )

    except Exception as e:

        # Don't crash the API if Databricks
        # is temporarily unavailable.
        print(
            f"Databricks warm-up failed: {e}"
        )


@app.on_event("startup")
def startup_event():

    # Stations are always available locally
    load_local_stations()

    # Databricks loads separately in background
    thread = threading.Thread(
        target=warm_up_databricks,
        daemon=True
    )

    thread.start()


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {
        "message": "MetroFlow API is running",
        "version": "1.1.0"
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "stations_loaded": bool(
            local_stations
        ),
        "network_loaded": (
            metro_graph is not None
        ),
        "crowd_cache_loaded": (
            _crowd_cache is not None
        )
    }


# ============================================================
# STATIONS
# ============================================================

@app.get("/stations")
def get_stations():

    load_local_stations()

    return {
        "stations": local_stations
    }


# ============================================================
# SINGLE ROUTE
# ============================================================

@app.get("/route")
def route(
    source: str,
    destination: str
):

    try:

        routes = find_multiple_routes(
            source,
            destination
        )

        result = routes[0]

        return {
            "source": source,
            "destination": destination,
            "travel_time_minutes":
                result[
                    "travel_time_minutes"
                ],
            "interchange_count":
                result[
                    "interchange_count"
                ],
            "lines_used":
                result[
                    "lines_used"
                ],
            "stations":
                result["stations"],
            "station_details":
                result[
                    "station_details"
                ]
        }

    except HTTPException:
        raise

    except Exception as e:

        print(
            f"Route error: {e}"
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "Route data is temporarily "
                "unavailable. Please try again."
            )
        )


# ============================================================
# SINGLE STATION CROWD
# ============================================================

@app.get("/crowd")
def crowd(
    station: str,
    hour: int
):

    if hour < 0 or hour > 23:

        raise HTTPException(
            status_code=400,
            detail=(
                "Hour must be "
                "between 0 and 23"
            )
        )

    try:

        cache = load_crowd_cache()

        station_key = (
            station.strip().lower()
        )

        row = next(
            (
                item
                for item in cache.values()
                if (
                    item["station_name"]
                    .strip()
                    .lower()
                    == station_key
                    and item["hour"] == hour
                )
            ),
            None
        )

        if row is None:

            raise HTTPException(
                status_code=404,
                detail=(
                    "Crowd data not found"
                )
            )

        return {
            "station":
                row["station_name"],
            "hour":
                row["hour"],
            "scheduled_train_visits":
                row[
                    "scheduled_train_visits"
                ],
            "activity_level":
                row["activity_level"],
            "crowd_score":
                row["crowd_score"],
            "crowd_level":
                row["crowd_level"],
            "estimated_seat_likelihood_pct":
                row[
                    "estimated_seat_likelihood_pct"
                ]
        }

    except HTTPException:
        raise

    except Exception as e:

        print(
            f"Crowd error: {e}"
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "Crowd data is temporarily "
                "unavailable. Please try again."
            )
        )


# ============================================================
# COMPLETE JOURNEY PLANNER
# ============================================================

@app.get("/plan")
def plan(
    source: str,
    destination: str,
    departure_time: str
):

    try:

        # Parse time
        departure_hour, departure_minute = (
            parse_departure_time(
                departure_time
            )
        )

        # Find routes from cached network
        raw_routes = find_multiple_routes(
            source,
            destination
        )

        final_routes = []

        # Crowd analysis
        for route in raw_routes:

            crowd_data = (
                get_crowd_for_route(
                    route["station_ids"],
                    departure_hour
                )
            )

            crowd_summary = (
                calculate_crowd_summary(
                    crowd_data
                )
            )

            route_result = {

                "travel_time_minutes":
                    route[
                        "travel_time_minutes"
                    ],

                "interchange_count":
                    route[
                        "interchange_count"
                    ],

                "lines_used":
                    route[
                        "lines_used"
                    ],

                "stations":
                    route["stations"],

                "station_details":
                    route[
                        "station_details"
                    ],

                "crowd_summary":
                    crowd_summary,

                "station_crowd":
                    crowd_data
            }

            final_routes.append(
                route_result
            )

        # Calculate scores
        for route in final_routes:

            route["route_score"] = (
                calculate_route_score(
                    route
                )
            )

        # Label routes
        final_routes = label_routes(
            final_routes
        )

        # Sort routes
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

        return {

            "source": source,

            "destination": destination,

            "departure_time":
                departure_time,

            "crowd_profile_hour":
                departure_hour,

            "routes":
                final_routes,

            "route_count":
                len(final_routes),

            "disclaimer":
                "Crowd and seat values are "
                "modelled estimates based on "
                "scheduled service activity and "
                "are not real-time occupancy or "
                "guaranteed seat availability."
        }

    except HTTPException:
        raise

    except Exception as e:

        print(
            f"Plan error: {e}"
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "Journey planning data is "
                "temporarily unavailable. "
                "Please try again."
            )
        )