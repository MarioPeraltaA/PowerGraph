"""Retrieve a microgrid from the electrical network.

Author::

    Mario Roberto Peralta. A.

"""

# Factory
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from dss import dss, enums, IDSS
import numpy as np
import matplotlib.pyplot as plt
import glob
# Patterns
import re
# Config files
import json
# GIS
import pandas as pd
import folium
import geopandas as gpd
from shapely.geometry import Point, LineString
# Set to False to leverage multiple contexts
dss.AllowChangeDir = False
dss.AllowForms = False
dss.AllowEditor = False


class Circuit(ABC):
    """Integrated circuit interface."""

    @abstractmethod
    def get_gis(
        self
    ):
        """Translate to map the electrical model."""
        ...

    @abstractmethod
    def get_geo_micro_grid(
        self
    ):
        """Withdraw microgrid's ``GIS`` from network."""
        ...

    @abstractmethod
    def put_micro_grid(
        self
    ):
        """Set microgrid's ``GIS`` to its electrical model."""
        ...


class AMI(ABC):
    """Abstract AMI device.

    Bear in mind ``LOCALIZACION`` is indeed the
    **NISE** and ``LOCALIZACION_REAL`` the actual location code.

    """

    @abstractmethod
    def load_data(self):
        """Read smart meters data."""
        ...

    @abstractmethod
    def set_df(self):
        """Process some datatype and columns name."""
        ...


@dataclass
class DSSCircuit(ABC):
    """The circuit electrical model.

    There is only one *Head* in case of monitor
    there is one of each kind ``Power`` and ``VI`` mode
    but nothing else.

    .. warning::

        Property :py:attr:`dss.Text.Command` when use ``compile``
        modifies the directory's root and sets the model's directory
        as the current one. So it is recommended to run it
        afterwards.

    .. warning::

        Avoid to use command :py:attr:`dss.ActiveCircuit.AllBusNames`
        as buses unique labels may come with *dots* and OpenDSS
        split nodes from bus name at first dot.

    """

    bus_coords_path: str = "./GIS-0709/0709_CoordinatesBuses*.csv"
    ckt_path: str = "./0709/0709_Master.dss"   # Master's directory
    solve_mode: int = enums.SolveModes.Daily
    control_mode: int = enums.ControlModes.Time
    algorithm: int = enums.SolutionAlgorithms.NormalSolve
    number: int = 96
    stepsize_min: int = 15
    bus_to_coord: dict[str, tuple[float, float]] = field(
        default_factory=dict
    )
    bus_to_transf: dict[str, str] | None = None
    bunches_df: pd.DataFrame | None = None
    ckt_data: dict[
        str, list[dict[str, str | list[tuple[float, float]]]]
    ] = field(
        default_factory=dict
    )
    buses: dict[str, tuple[float, float]] = field(default_factory=dict)
    head_meter: str = ""
    head_monitors: list[str, str] = field(default_factory=list)   # PWR, VI
    dss: IDSS = field(init=False)

    def __post_init__(
            self
    ):
        """Set and solve circuit."""
        self.load_bus_coords()          # Geometry data
        self.dss = dss.NewContext()     # Multiple engine threads
        self.load_ckt(to_solve=True)    # Electrical model
        self.add_head_meter()           # Topology
        self.add_head_monitor(mode=enums.MonitorModes.Power)
        self.add_head_monitor(mode=enums.MonitorModes.VI)
        # Solve again to update
        self.dss.ActiveCircuit.Solution.Solve()
        try:
            if not self.dss.ActiveCircuit.Solution.Converged:
                raise RuntimeError("Not optimal solution found.")
        except RuntimeError as e:
            message = (
                f"NoConvergence: {e}"
            )
            print(message)
        else:
            self.set_ckt_data()           # Elements properties
            self.check_geometry()         # Missing buses's locations

    def load_bus_coords(
            self
    ):
        """Read and map each bus name to its geographical position."""
        bus_paths = glob.glob(self.bus_coords_path)   # Both MV and LV
        for path in bus_paths:
            with open(path, "r") as file:
                for row in file:
                    if row:
                        bus_id, northing_y, easting_x = row.split(",")
                        self.bus_to_coord[bus_id.lower()] = (
                            float(northing_y), float(easting_x)
                        )

    def key_coordinates(
            self,
            bus_id: str,
            add_bus: bool = True
    ) -> tuple[float, float] | None:
        """Index bus coordinates.

        Remove OpenDSS node suffix (one or more '.<int>'
        segments at the end). Handles arbitrary number
        of nodes: .1, .1.2, .1.2.3.0, etc.
        Preserves dots inside the true bus name.

        ``BUSLVn494464.97098_1099749.62864pLVpCN.1.2.3``

        to

        ``BUSLVn494464.97098_1099749.62864pLVpCN``

        If ``add_bus`` is True then the given buses
        of the element are internally stored at
        :py:attr:`gdss.DSSCircuit.buses`

        """
        bus_name = re.sub(r'(\.\d+)+$', '', bus_id)   # Strip nodes
        bus_name = bus_name.lower()
        try:
            coords = self.bus_to_coord[bus_name]
        except KeyError as e:
            # print(f"BusNotFound: {e}")  <-- see `check_geometry()` method
            return
        else:
            if add_bus:
                self.buses[bus_name] = coords
            return coords

    def item_buses_location(
            self,
            buses_id: list[str]
    ) -> list[tuple[float, float]] | None:
        """Map each buses of a divice to its geographical position.

        .. note::

            The ``Transformer`` class may not have
            location to all of its buses as long
            as any of them has it that one will
            be used as ``Transformer`` is seen as a
            Point geometry.

        """
        bus_location: list[tuple[float, float]] = []
        for bus_id in buses_id:
            coords = self.key_coordinates(bus_id)
            if coords:
                if coords in bus_location:
                    continue
                else:
                    bus_location.append(coords)
        if bus_location:
            return bus_location

    def put_daily_solution_mode(
            self
    ):
        """Type of solution."""
        # Retrieve context interfaces
        dssSolution = self.dss.ActiveCircuit.Solution

        # Set kind of solution
        dssSolution.Mode = self.solve_mode
        dssSolution.ControlMode = self.control_mode
        dssSolution.Number = self.number
        dssSolution.StepsizeMin = self.stepsize_min
        dssSolution.Algorithm = self.algorithm

    def load_ckt(
            self,
            to_solve: bool = False
    ):
        """Load and solve network."""
        # Compile a model
        self.dss.Text.Command = f'compile "{self.ckt_path}"'
        self.put_daily_solution_mode()
        if to_solve:
            self.dss.ActiveCircuit.Solution.Solve()

    def set_ckt_data(
            self
    ):
        """Orginize circuit devices and parameters.

        Retrain all data of the electrical model by retreving
        all parameters values of each element of the ciruict.

        .. warning::

            This values are taking as string datatypes
            so for further data or geometry analysis
            make sure each field datatype is the proper one.

        """
        elements = self.dss.ActiveCircuit.AllElementNames

        for device in elements:
            if device:
                fields = {}
                fields["dssname"] = device.lower()
                dss_element = (
                    self.dss.ActiveCircuit.ActiveCktElement(device)
                )
                fields["location"] = self.item_buses_location(
                    dss_element.BusNames
                )

                # Electrical properties
                parameters = dss_element.AllPropertyNames
                for ft in parameters:
                    val: str = dss_element.Properties(ft).Val
                    fields[ft] = val
                layer = (
                    self.dss.ActiveCircuit.ActiveClass.ActiveClassName
                )

                if layer in self.ckt_data:
                    self.ckt_data[layer].append(fields)
                else:
                    self.ckt_data[layer] = [fields]

    def check_geometry(
            self
    ):
        """Reveal elements with missing geographical position.

        List those devices in the electrical model whose
        location was not found.

        .. note::

            The class element ``Vsource`` whose default name is
            ``source`` at bus ``sourcebus`` it is internally
            set by *OpenDSS* so it may not have location.

        .. warning::

            Name of third floating winding of a three-phase transformer
            specified by the user e.g. ``BUST_0709`` may be
            replaced behind the scene by *OpenDSS* to
            the generic name ``hvmv_3``.

        """
        phantom_objs: dict[str, list[str]] = {}

        for devices in self.ckt_data.values():
            for obj in devices:
                if obj['location']:
                    continue
                else:
                    element = (
                        self.dss.ActiveCircuit.ActiveCktElement(obj['dssname'])
                    )
                    phantom_objs[obj['dssname']] = element.BusNames

        try:
            if phantom_objs:
                raise KeyError("Location of bus used in model is missing.")
        except KeyError as e:
            logg = (
                f"PositionNotFound: {e}\n"
                f"{phantom_objs}"
            )
            print(logg)
        else:
            return

    def add_meter(
            self,
            full_name_element: str = "transformer.substation",
            meter_id: str = "feeder_meter",
            terminal: int = 1,
            option: str = "(E, R, C)"
    ) -> str:
        """Instantiate and set a single EnergyMeter."""
        self.dss.Text.Command = (
            f"New EnergyMeter.{meter_id} "
            f"element={full_name_element} "
            f"terminal={terminal} "
            f"option={option}"
        )
        return meter_id

    def add_monitor(
            self,
            full_name_element: str = "transformer.substation",
            monitor_id: str = "substation_monitor_1",
            terminal: int = 1,
            mode: int = enums.MonitorModes.Power,
            polar: bool = False
    ) -> str:
        """Instantiate and set a single monitor."""
        dssMonitors = self.dss.ActiveCircuit.Monitors
        monitors_id = dssMonitors.AllNames
        if monitor_id in monitors_id:
            return monitor_id

        if mode == enums.MonitorModes.VI:
            notation = "VIpolar"
        else:
            notation = "ppolar"
        self.dss.Text.Command = f"new monitor.{monitor_id} {notation}={polar}"
        dssMonitors.Name = monitor_id
        dssMonitors.Element = full_name_element
        dssMonitors.Terminal = terminal
        dssMonitors.Mode = mode
        return monitor_id

    def add_head_meter(
            self,
            source_bus_id: str = "sourcebus",
            terminal: int = 1,
    ):
        """Embed EnergyMeter right at feeder's head.

        To assess Topology analysis and collect global Registers.

        Raises
        ------
        TypeError
            As only and solely one PDE must be connected
            to this sourcebus.

        ValueError
            Floating sourcebus.

        .. warning::

            The command :py:attr:`iBus.AllPDEatBus` it is not reliable
            as may return branches whose Bus has not connections
            at all.

        """
        ibus_obj = self.dss.ActiveCircuit.ActiveBus(source_bus_id)
        pd_elements = ibus_obj.AllPDEatBus
        # Full name branches
        pd_elements = self.dss.ActiveCircuit.ActiveBus.AllPDEatBus
        pd_elements = [
            None if (e) and (e.lower() in {"none", "nan", "null"}) else e
            for e in pd_elements
        ]
        # Kick out falsy items
        feeder_branches = list(filter(None, pd_elements))
        # Add meter
        try:
            if feeder_branches:
                n_branches = len(feeder_branches)
                if n_branches != 1:
                    raise TypeError("Multiple branches at head.")
            else:
                raise ValueError("No PDE at feeder.")
        except ValueError as e:
            logg = (
                f"EmptyBranches: {e}"
            )
            print(logg)
        except TypeError as e:
            logg = (
                f"NonUniqueHead: {e}"
            )
        else:
            branch = feeder_branches[0]
            _ = self.dss.ActiveCircuit.SetActiveElement(branch)
            element_id = self.dss.ActiveClass.Name
            meter_id = self.add_meter(
                branch, f"{element_id}_meter", terminal
            )
            self.head_meter = meter_id

    def add_head_monitor(
            self,
            source_bus_id: str = "sourcebus",
            terminal: int = 1,
            mode: int = enums.MonitorModes.Power
    ):
        """Deploy monitors to first PDE connected to sourcebus.

        To keep an eye on external network modeled as
        the main circuit source.

        Raises
        ------
        TypeError
            As only and solely one PDE must be connected
            to this sourcebus.

        ValueError
            Floating sourcebus.

        .. warning::

            The command :py:attr:`iBus.AllPDEatBus` it is not reliable
            as may return branches whose Bus has not connections
            at all.

        """
        ibus_obj = self.dss.ActiveCircuit.ActiveBus(source_bus_id)
        pd_elements = ibus_obj.AllPDEatBus
        # Full name branches
        pd_elements = self.dss.ActiveCircuit.ActiveBus.AllPDEatBus
        pd_elements = [
            None if (e) and (e.lower() in {"none", "nan", "null"}) else e
            for e in pd_elements
        ]
        # Kick out falsy items
        feeder_branches = list(filter(None, pd_elements))
        # Add monitor, check for unique item
        try:
            if feeder_branches:
                n_branches = len(feeder_branches)
                if n_branches != 1:
                    raise TypeError("Multiple branches at head.")
            else:
                raise ValueError("No PDE at feeder.")
        except ValueError as e:
            logg = (
                f"EmptyBranches: {e}"
            )
            print(logg)
        except TypeError as e:
            logg = (
                f"NonUniqueHead: {e}"
            )
        else:
            branch = feeder_branches[0]   # Full name without nodes
            _ = self.dss.ActiveCircuit.SetActiveElement(branch)
            element_id = self.dss.ActiveClass.Name
            monitor_id = self.add_monitor(
                branch, f"{element_id}_monitor_{mode}", terminal, mode
            )
            self.head_monitors.append(monitor_id)

    def get_meter_data(
            self,
            meter_id: str = "substation_meter",
            register_i: int = enums.EnergyMeterRegisters.ZoneLosseskWh
    ) -> float:
        """Retrieve requested Register value from EnergyMeter.

        .. warning::
            EnergyMeter Registers are neither clear up
            nor reset after getting its data.

        """
        dssMeters = self.dss.ActiveCircuit.Meters
        dssMeters.Name = meter_id
        return dssMeters.RegisterValues[register_i]

    def get_monitor_data(
            self,
            monitor_id: str = "feeder_pq",
            reset: bool = True
    ) -> np.ndarray:
        """Key and retrieve monitor's data.

        Active Circuit must be run already.

        """
        dssMonitors = self.dss.ActiveCircuit.Monitors
        # Activete monitor element
        dssMonitors.Name = monitor_id

        try:
            if self.dss.ActiveCircuit.Solution.Converged:
                if dssMonitors.Name == monitor_id:
                    # Retrieve data
                    monitor_data = dssMonitors.AsMatrix()
                    if reset:
                        dssMonitors.Reset()   # Reset only active one
                else:
                    raise ValueError(f"Monitor {monitor_id} not found")

            else:
                raise RuntimeError(
                    f"Circuit {self.dss.ActiveCircuit.Name} did not converge"
                )
        except RuntimeError as e:
            print(f"MaxIterReached: {e}.")
            return
        except ValueError as e:
            print(f"ElementNotFound: {e}.")
            return

        return monitor_data

    def voltage_zones(
            self
    ) -> dict[str, list[tuple[str, float]]]:
        """Match each transformer's buses to its voltage level."""
        itransf = self.dss.ActiveCircuit.Transformers
        transf_ids = itransf.AllNames
        zones: dict[str, list[tuple[str, float]]] = {}
        for id_ in transf_ids:
            itransf.Name = id_  # Activates element also higher level
            ifts = self.dss.ActiveCircuit.ActiveCktElement.Properties
            buses = ifts("Buses").Val
            wng_nodes = [
                b.strip() for b in buses.strip("[]").split(",") if b.strip()
            ]
            wng_buses = [
                re.sub(r'(\.\d+)+$', '', b) for b in wng_nodes
            ]
            kvs = ifts("kVs").Val
            wng_kvolts = [
                float(v) for v in kvs.strip("[]").split(",") if v.strip()
            ]
            zones[id_] = [(b, v) for b, v in zip(wng_buses, wng_kvolts)]
        return zones

    def get_voltage_zone(
            self,
            kvoltage_level: float = 1.0
    ) -> dict[str, list[str]]:
        """Retrieve all downstream transformers.

        Given a voltage level e.g. ``1.0 kV``
        filter out all upstream transformers above this level.
        Create another struture to index transformer by its bus.

        """
        zones = self.voltage_zones()
        roots: dict[str, list[str]] = {}    # Roots and their parents (hats)
        bus_to_transf = {}
        for id_, buses_kv in zones.items():
            bus_id, kv = buses_kv[-1]       # Last winding bus
            if kv < kvoltage_level:
                roots[bus_id] = [
                    vertex for vertex, _ in buses_kv[:-1] if vertex != bus_id
                ]
                bus_to_transf[bus_id] = id_
        self.bus_to_transf = bus_to_transf
        return roots

    def map_loads(
            self,
            *load_path: str
    ) -> dict[str, int]:
        """Map load name to location code."""
        locations_id: dict[str, int] = {}

        for path in load_path:
            with open(path, "r") as file:
                for sentence in file:
                    sentence = sentence.strip().lower()
                    if "!loc=" in sentence:
                        load_id: str = sentence.split(" ")[1]  # dss full name
                        loc: str = (
                            sentence.split("!loc=")[-1]
                            .strip()
                            .replace("-", "")
                        )
                        if loc:
                            try:
                                loc: int = int(loc)
                            except ValueError as e:
                                message: str = (
                                    f"NoLocationCode: {e} "
                                    f"of element '{load_id}'."
                                )
                                print(message)
                                continue
                            else:
                                locations_id[load_id] = loc
        return locations_id

    def write_json(
            self,
            path: str,
            data: dict
    ):
        """Write out data as json file."""
        with open(
            path, mode="w", encoding="utf-8"
        ) as file:
            json.dump(data, file, indent=4, ensure_ascii=False)

    def get_bunches_df(
            self,
            bunches_path: str = "./City/bunches.json",
            loads_loc_path: str = "./City/loadsloc.json"
    ) -> pd.DataFrame:
        r"""Hold data of bunches (group of loads).

        Build efficient data structure to access group
        of loads data. This data is:
        Transformer name and its capacity in kVA
        where costumer (meter) it is connected to
        as well as load location code and its
        average apparent power :math:`\vec{S}` as [kW, kVAr].

        """
        # Read bunches and load location as dict
        with open(bunches_path, "r") as file:
            bunches = json.load(file)

        with open(loads_loc_path, "r") as file:
            loads_loc = json.load(file)

        # Explode clusters
        rows: list[tuple, int] = []
        for transfr, loads in bunches.items():
            for load in loads:
                load_name = load.lower()
                if load_name in loads_loc:
                    transfr_elem = (
                        self.dss.ActiveCircuit
                        .ActiveCktElement(f"transformer.{transfr}")
                    )
                    ifts = transfr_elem.Properties
                    kva_str = ifts("kVA").Val
                    load_elem = (
                        self.dss.ActiveCircuit
                        .ActiveCktElement(load_name)
                    )
                    ifts = load_elem.Properties
                    meter_id = ifts("Daily").Val
                    ishape = self.dss.ActiveCircuit.LoadShapes
                    ishape.Name = meter_id
                    kw_avg, kvar_avg = (
                        np.average(ishape.Pmult), np.average(ishape.Qmult)
                    )
                    load_kva = np.sqrt(kw_avg**2 + kvar_avg**2)
                    try:
                        kva = float(kva_str)
                    except ValueError as e:
                        print(f"kVA capacity {e} of transformer {transfr}.")
                        continue
                    else:
                        rows.append(
                            (transfr,
                             kva,
                             load_name,
                             loads_loc[load_name],
                             kw_avg,
                             kvar_avg,
                             load_kva)
                        )
        # Instantiate DataFrame
        columns: list[str] = [
            "Transformer",
            "kVA",          # Transformer capacity
            "Load",
            "Location",     # Load location code
            "Pavg",
            "Qavg",
            "Savg"          # Apparent power kVA
        ]
        bunches_df = pd.DataFrame(
            rows, columns=columns
        )
        self.bunches_df = bunches_df
        return bunches_df

    def transformer_loading_pct(
            self
    ) -> pd.DataFrame:
        """Compute how loaded a transformer is."""
        bunches_df = self.bunches_df.copy()
        gr = bunches_df.groupby("Transformer")
        # Add up groups loading
        agg = gr.agg(
            Ptot=("Pavg", "sum"),
            Qtot=("Qavg", "sum"),
            kVA=("kVA", "first")   # All rows in a group have same kVA
        ).reset_index()
        # Compute apparent power and then loading as percentage
        agg["S"] = np.sqrt(agg['Ptot']**2 + agg['Qtot']**2)
        agg["Loading"] = 100.0 * agg['S'] / agg['kVA']
        return agg


@dataclass
class GISCircuit(ABC):
    """Handle and cope with GIS (No electrical modeling)."""

    ckt: DSSCircuit
    gis_data: dict[str, pd.DataFrame] = field(     # Without buses
        default_factory=dict
    )
    layers: dict[str, gpd.GeoDataFrame] = field(   # With buses
        default_factory=dict
    )
    brush: list[str] = field(default_factory=list)

    def __post_init__(
            self
    ):
        """Build up *GIS*."""
        self.set_geometries()
        self.build_layers()
        self.add_buses_layer()
        self.paint_map()

    def set_geometries(
            self
    ):
        """Instantiate *GIS* geometry.

        Convert tuple of floats to actual geometry
        data type of elements and circuit devices (no buses).

        """
        data = self.ckt.ckt_data

        for layer, ckt_devices in data.items():
            gis_devices = []
            for obj in ckt_devices:
                fields = {**obj}   # Independent copy of dict
                coords = fields['location']
                if coords:
                    # Points
                    if len(coords) == 1:
                        fields['location'] = Point(coords[0])
                    # Strings
                    else:
                        fields['location'] = LineString(coords)
                # Empty geometry
                else:
                    fields['location'] = Point()
                gis_devices.append(fields)

            self.gis_data[layer] = pd.DataFrame(gis_devices)

    def get_buses_geom(
            self
    ) -> list[dict[str, str | Point]]:
        """Add both MV and LV buses Points geometries.

        Circuit :py:class:`gdss.DSSCircuit` either converged
        or not should be already run and solve tried.

        """
        buses = self.ckt.buses
        buses_data: list[dict[str, str | Point]] = []
        for bus_name, coords in buses.items():
            if bus_name:
                if coords:
                    point_geom = Point(coords)
                else:
                    point_geom = Point()
                bus_data = {
                    "dssname": bus_name,
                    "location": point_geom
                }
                buses_data.append(bus_data)

        return buses_data

    def add_buses_layer(
            self,
            crs: str = "EPSG:5367"
    ):
        """Instantiate :py:class:`gpd.GeoDataFrame` of Bus object."""
        buses_data = self.get_buses_geom()
        buses_gdf = gpd.GeoDataFrame(
            buses_data, geometry='location', crs=crs
        )
        self.layers['Bus'] = buses_gdf

    def build_layers(
            self,
            crs: str = "EPSG:5367"
    ):
        """Create GeoDataFrame of circuit Elements.

        By ``Element`` means a either PCE or PDE or
        circuit devices (Fuses, Meters, Monitors, Switches)
        but Buses.

        """
        for layer, devices in self.gis_data.items():
            gdf = gpd.GeoDataFrame(devices, geometry="location", crs=crs)
            self.layers[layer] = gdf

    def paint_map(
            self,
            seed: int = 7859
    ) -> dict[str, list[str, str]]:
        """Assign eye-cathing color to each layer.

        Uses ``rng.shuffle`` instead of ``rng.integers`` to make sure
        all colors are different.

        """
        # Get the list of X11/CSS4 color names
        lib_colors = list(plt.cm.colors.cnames)
        size = len(self.layers)
        # Seed for reproducibility
        rng = np.random.default_rng(seed=seed)
        rnd_ints = np.arange(0, len(lib_colors))
        rng.shuffle(rnd_ints)

        for c in rnd_ints[:size]:
            self.brush.append(lib_colors[c])

    def explore_ckt(
            self
    ) -> folium.Map:
        """Pile up layers."""
        ckt_map = folium.Map(
            crs="EPSG3857",
            control_scale=True,
            tiles="CartoDB.PositronNoLabels",
            zoom_start=15
        )
        for c, (layer, gdf) in enumerate(self.layers.items()):
            # Skip empty GeoSeries
            if not all(gdf['location'].is_empty):
                ckt_map = gdf.explore(
                    m=ckt_map,
                    name=layer,
                    show=False,
                    color=self.brush[c],
                    popup=True
                )

        # Customize tile
        (
            folium
            .TileLayer("Cartodb dark_matter", show=False)
            .add_to(ckt_map)
        )
        folium.LayerControl().add_to(ckt_map)
        return ckt_map


@dataclass()
class CktGraph(ABC):
    """Skeleton factory."""

    ckt: DSSCircuit
    full_net: bool = False
    vertices: list[str] = field(default_factory=list)
    edges: list[tuple[str, str]] = field(default_factory=list)
    adj: dict[str, list[str]] = field(default_factory=dict)
    microgrid: list[tuple[str, str]] | None = field(
        default_factory=list
    )

    def __post_init__(self):
        """Graph representation of Network."""
        self.set_adjacency_list()

    def add_vertex(
        self,
        vertex_id: str
    ) -> str:
        """Instantiate vertex.

        Vertex is a unique bus of the circuit in spite
        of its nodes.

        .. Note::
            In opendss a Bus may have multiple Nodes.

        """
        if vertex_id not in self.vertices:
            self.vertices.append(vertex_id)

    def add_edge(
        self,
        from_vertex: str,
        to_vertex: str
    ):
        """Instantiate edge.

        Edge is a branch with two ends. i.e. Connection
        between two vertices.

        """
        self.add_vertex(from_vertex)
        self.add_vertex(to_vertex)
        edge = (from_vertex, to_vertex)
        if edge not in self.edges:
            self.edges.append(edge)

    def untwist_branch(
            self,
            buses: list[str],
    ):
        """Cope with odd branches.

        Odd branches (more than two ends/terminals) such
        as three phase three winding transformer are taken
        as a cycle graph third order :math:`C_{3}`.

        """
        branches = [
            (buses[i], buses[(i + 1) % len(buses)]) for i in range(len(buses))
        ]
        for edge in branches:
            if "hvmv_3" in edge:
                continue
            self.add_edge(edge[0], edge[1])

    def head_zone(
            self
    ) -> list[str]:
        """Retrieve all branches seen by head meter.

        This does not imply the circuit's zone it is
        connected all the way as a single zone may still
        have islands.

        .. note::

            If there is only one meter and such one it is the
            *head meter* then in case the network it is
            disconnected those other components would be out of
            the *head meter*'s range.

        """
        iMeters = self.ckt.dss.ActiveCircuit.Meters
        iMeters.Name = self.ckt.head_meter
        return iMeters.AllBranchesInZone

    def collect_branches(
            self
    ):
        """Gather all Power Delivery Elements (PDE)."""
        return self.ckt.dss.ActiveCircuit.PDElements.AllNames

    def build_graph(
            self,
    ):
        """Generate undirected graph."""
        if self.full_net:
            branches = self.collect_branches()
        else:
            branches = self.head_zone()

        dssCircuit = self.ckt.dss.ActiveCircuit
        for edge in branches:
            if edge:
                dssBranch = dssCircuit.ActiveCktElement(edge)
                nodes = dssBranch.BusNames
                # Strip specific nodes
                buses = [re.sub(r'(\.\d+)+$', '', node) for node in nodes]
                if len(set(buses)) != 2:
                    self.untwist_branch(buses)
                else:
                    self.add_edge(buses[0], buses[1])

    def set_adjacency_list(
            self
    ):
        """Graph representation."""
        self.build_graph()
        self.adj = {
            v: [] for v in self.vertices
        }
        for edge in self.edges:
            self.adj[edge[0]].append(edge[1])
            self.adj[edge[1]].append(edge[0])

    def get_graph(
            self,
            edges: list[tuple[str, str]]
    ):
        """Turn list of edges into dict adj."""
        nodes: list = [end for ends in edges for end in ends]
        graph: dict[str, list[str]] = {
            n: [] for n in nodes
        }
        for edge in edges:
            graph[edge[0]].append(edge[1])
            graph[edge[1]].append(edge[0])
        return graph

    def dfs_edges(
            self,
            graph: dict[str, list[str]],
            root_bus: str | None = "sourcebus",
            depth_limit: int | None = None
    ):
        """Depth First Search.

        To traverse graph. If ``root`` (source) is provided
        then yield only edges in the component reachable
        from source. This pattern mimics `networkX <https://networkx.org/>`_.
        See [1]_ and [2]_.

        References
        ----------
        .. [1] http://www.ics.uci.edu/~eppstein/PADS
        .. [2] https://en.wikipedia.org/wiki/Depth-limited_search


        .. note::

            The ``root`` is not necessary the *bus head* of the
            electrical network.

        """
        if root_bus is None:
            # Edges for all components
            vertices = list(graph.keys())
        else:
            # Edges for components with source
            vertices = [root_bus]

        if depth_limit is None:
            depth_limit = len(graph)

        visited = set()
        for start in vertices:
            if start in visited:
                continue
            visited.add(start)
            stack = [(start, graph[start])]
            depth_now = 1
            while stack:
                parent, children = stack[-1]
                for child in children:
                    if child not in visited:
                        # Discovered edge
                        yield parent, child
                        visited.add(child)
                        if depth_now < depth_limit:
                            # Add child and grandchildren to stack
                            stack.append((child, graph[child]))
                            depth_now += 1
                            break
                else:
                    _ = stack.pop()
                    depth_now -= 1

    def power_dfs(
            self,
            graph: dict[str, list[str]],
            root_bus: str = "lv_bus",
            hats: list[str] = ["hv_bus, mv_bus"]
    ):
        """Depth First Search.

        To traverse graph. If ``root`` (source) is provided
        then yield only edges in the component reachable
        from source. This pattern mimics `networkX <https://networkx.org/>`_.
        See [1]_ and [2]_.

        References
        ----------
        .. [1] http://www.ics.uci.edu/~eppstein/PADS
        .. [2] https://en.wikipedia.org/wiki/Depth-limited_search


        .. note::

            The ``root`` is not necessary the *bus head* of the
            electrical network.

        """
        vertices = [root_bus]
        visited = set(hats)
        for start in vertices:
            if start in visited:
                continue
            visited.add(start)
            stack = [(start, graph[start])]
            depth_now = 1
            while stack:
                parent, children = stack[-1]
                for child in children:
                    if child not in visited:
                        # Discovered edge
                        yield parent, child
                        visited.add(child)
                        # Add child and grandchildren to stack
                        stack.append((child, graph[child]))
                        depth_now += 1
                        break
                else:
                    _ = stack.pop()
                    depth_now -= 1

    def is_connected(
            self,
            graph: dict[str, list[str]]
    ) -> bool:
        """Verify if graph is connected."""
        vertices: list[str] = list(graph.keys())
        edges = self.dfs_edges(
            graph=graph,
            root_bus=vertices[0],
            depth_limit=None
        )
        sub_vertices = {end for ends in edges for end in ends}
        return len(sub_vertices) == len(vertices)

    def retrieve_subgraph(
            self,
            start_up_vertices: list[tuple[str, int]],
    ) -> list[tuple[str, str]]:
        """Select subgraph.

        Return a subgraph given multiples starting vertices
        let's call them *anchor buses*.

        Parameters
        ----------
        start_up_vertices : list[tuple[str, int]]
            Top nodes of each graph path as pairs
            ``(root_bus, depth_limit)``

        .. warning::

            Make sure depth levels are good enough so all
            subgraphs end up touching each other resulting
            in a connected one.

        """
        for root, depth in start_up_vertices:
            branches = self.dfs_edges(self.adj, root, depth)
            for edge in branches:
                if edge not in self.microgrid:
                    self.microgrid.append(edge)

        micro_graph = self.get_graph(self.microgrid)

        try:
            if not self.is_connected(micro_graph):
                raise Exception("Network has multiple components")
        except Exception as e:
            logg: str = (
                f"DisconnectedGraph: {e}. You may go deeper."
            )
            print(logg)
        finally:
            return self.microgrid

    def write_ckt(
            self,
            nodes: list[str],
            ckt_directory: str = "./0709/*.dss",
            output_directory: str = "./City/dssmodel/"
    ):
        """Write out dss script circuit model.

        Pass the whole integrated *.dss* files of the
        circuit.

        """
        dss_files = glob.glob(ckt_directory)

        for input_script in dss_files:
            dss_script: list[str] = []
            with open(input_script, "r") as dssfile:
                for row in dssfile:
                    row = row.lower()
                    for bus in nodes:
                        if bus in row:
                            if row not in dss_script:
                                dss_script.append(row)
                            break
                    else:
                        continue
            # Output script
            micro_ckt = "\n".join(dss_script)

            file = input_script.split("/")[-1]
            output_path = f"{output_directory}{file}"
            with open(output_path, "w") as dssfile:
                dssfile.write(micro_ckt)

    def get_bunches(
            self,
            kvoltate_zone: float = 1.0
    ) -> dict[str, list[str]]:
        """Cluster loads by transformers.

        Traverse down stream the graph using DFS
        argorithm whose root is the bus where the last widing it is
        conntected to.
        Then gather all buses that hold a loads (fruit) in current walk train.
        Finally Associate current transformer to this loads
        and call it *bunch*.

        """
        roots = self.ckt.get_voltage_zone(kvoltate_zone)
        bus_to_transf = self.ckt.bus_to_transf
        bunches: dict[str, list[str]] = {}
        for root, hats in roots.items():
            train = self.power_dfs(self.adj, root, hats)
            vertices = list({end for ends in train for end in ends})
            # Gather all the fruits (loads) from the train
            loads = []
            for vertex in vertices:
                ibus = self.ckt.dss.ActiveCircuit.ActiveBus(vertex)
                fruits: list[str] = list(filter(None, ibus.LoadList))
                if fruits:
                    loads += fruits

            bunches[bus_to_transf[root]] = list(set(loads))
        return bunches
