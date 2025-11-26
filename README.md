
# **Electrical Distribution Network Traversal and Microgrid Extraction Framework**

> This repository provides a modular and extensible framework for **analyzing, traversing, and extracting sub-networks** from distribution systems modeled in **OpenDSS**. It combines electrical modeling, GIS processing, and graph-theoretic tools to support research tasks such as:

* **Automatic retrieval of microgrids** from a feeder.
* **Connectivity analysis** and detection of inconsistencies in the topology.
* **Mapping OpenDSS circuits to GIS geometries** and building interactive geographic visualizations.
* **Grouping customer loads by their supplying transformer**.
* **Traversing distribution networks** using depth-first search (DFS).
* **Constructing graph representations** of circuits based on buses and branches.
* **Integration of AMI data** and geospatial information.

The library is designed with an academic style and emphasizes clarity, explicitness, and ease of experimentation for power-system researchers.

Author: Mario R. Peralta A. <br>
Contact: mario.peralta@ieee.org

---

## 1. Overview

Modern distribution-system studies often require combining:

1. **Electrical models** (OpenDSS)
2. **Geospatial data** (GIS shapefiles or CSV coordinates)
3. **Graph algorithms** (topology traversal, microgrid extraction)
4. **Smart-meter (AMI) data streams**

This framework provides unified abstractions for these domains.
At its core, three major components are defined:

### `DSSCircuit`

A high-level wrapper around an OpenDSS model that:

* Loads and solves the circuit.
* Retrieves electrical properties of all devices.
* Maps OpenDSS buses to GIS coordinates.
* Automatically attaches meters and monitors at feeder heads.
* Collects geometry and elemental metadata into structured dictionaries.
* Identifies objects with missing or inconsistent locations.

### `GISCircuit`

A GIS representation of the electrical network:

* Converts device coordinates into **shapely** geometries.
* Produces **GeoDataFrame** layers for buses, lines, transformers, fuses, switches, etc.
* Assigns map colors deterministically.
* Creates fully interactive web maps using **folium**.

### `CktGraph`

A graph-theoretic model of the distribution system:

* Vertices correspond to **unique buses**.
* Edges correspond to **branches (PDEs/PCEs)**.
* Supports odd branches such as 3-winding transformers.
* Provides DFS traversal suitable for:

  * microgrid extraction,
  * reachability analysis,
  * topology validation,
  * graph cleaning and component detection.

---

## 2. Key Capabilities

### **2.1 Microgrid Retrieval**

The framework reconstructs the subgraph reachable from a set of “pivot” buses (e.g., a local generator, a boundary switch, or a protection device).
This allows:

* Study of islanding strategies
* Microgrid impact assessment
* Load aggregation within a microgrid boundary
* Exporting the resulting microgrid as a new OpenDSS model

### **2.2 Connectivity and Topology Cleaning**

The system detects:

* Buses referenced in the OpenDSS model but missing coordinates
* Elements with ill-defined terminals
* Inconsistent node labels (e.g., multiple `.1.2.3` suffixes)
* Unconnected components
* Nodes that OpenDSS silently renames (e.g., 3rd transformer winding → `hvmv_3`)

The graph structure facilitates:

* Detection of isolated clusters
* Identification of electrically unreachable elements
* Verification of feeder integrity

### **2.3 GIS Integration**

Features include:

* Conversion of all circuit geometry to **point** and **linestring** data
* Building multi-layer interactive maps
* Automatic color assignment
* Inclusion of both MV and LV buses
* Transformer-to-load spatial grouping

This is essential for:

* Visual diagnostics
* Data validation
* Field asset matching
* Spatial load studies

### **2.4 AMI Data Processing**

An abstract base class (`AMI`) defines a standard interface for:

* Loading smart meter datasets
* Harmonizing formats and datatypes
* Mapping load locations to network components

This makes the framework suitable for:

* Customer clustering
* Daily/annual load shape analysis
* Consumption-profile spatial aggregation

---

## 3. Software Architecture

### **3.1 Abstract Base Classes**

The system uses ABCs to define clear contracts:

* `Circuit`: defines GIS–electrical model interoperability
* `AMI`: defines smart meter ingestion
* `CktGraph`: defines graph operations

### **3.2 DSSCircuit Core Behaviors**

* Uses independent OpenDSS context (`dss.NewContext`)
* Auto-solves the circuit and enforces daily simulation settings
* Parses every element and its properties
* Uses regex to robustly strip node suffixes from bus names
* Handles odd-terminal transformers via cycle-graph construction

### **3.3 GIS Processing**

* Every device becomes a Shapely geometry
* GeoDataFrames retain all property metadata
* Buses are included as a dedicated layer

### **3.4 Graph Construction**

* All PDEs across meter zones are included
* DFS handles depth-limiting, full traversal, and component selection
* Graph can be exported or used for microgrid extraction

---

## 4. Typical Use Cases

### **4.1 Extract a microgrid**

```python
ckt = DSSCircuit()
graph = CktGraph(ckt)
mg_edges = list(graph.dfs_edges(graph.adj, root_bus="local_gen_bus"))
```

### **4.2 Visualize entire circuit**

```python
gis = GISCircuit(ckt)
m = gis.explore_ckt()
m.save("ckt_map.html")
```

### **4.3 Group loads by transformer**

```python
groups = group_loads_by_transformer(ckt.ckt_data)
```

### **4.4 Check consistency of bus geometry**

```python
ckt.check_geometry()
```

---

## 5. Folder Structure

```
.
├── dss/                      # OpenDSS interface
├── gis/                      # GIS data and layers
├── ami/                      # AMI ingestion
├── graph/                    # Graph and traversal tools
├── examples/                 # Usage examples and notebooks
└── README.md                 # This document
```

---

## 6. Dependencies

* Python ≥ 3.10
* dss-python
* numpy
* pandas
* geopandas
* shapely
* folium
* matplotlib

---

## 8. License

MIT License.
Feel free to use, adapt, and extend the code.


> **Peralta A., M. R.**
*Traversal and Microgrid Extraction in Georeferenced Distribution Networks.* 2025.