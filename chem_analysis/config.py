import enum


class PlottingLibraries(enum.Enum):
    PLOTLY = 0
    MATPLOTLIB = 1
    PYGRAPHQT = 2


class Configuration:
    plotting_libraries = PlottingLibraries

    def __init__(self):
        self.preferred_plot = PlottingLibraries.PLOTLY
        self._plotting_libraries = []
        self.sig_fig = 3
        self.table_format = "rounded_grid"

        self._find_available_plotting_libraries()

    def load_from_env(self):
        pass  # TODO: add support

    def get_plotting_options(self) -> list[PlottingLibraries]:
        self._find_available_plotting_libraries()
        if self._plotting_libraries is None:
            raise RuntimeError("No plotting libraries installed. Please install one of the following:"
                               "\n\tplotly: `pip install plotly'"
                               "\n\tmatplotlib: 'pip install matplotlib'"
                               "\n\tpygraphqt: 'pip install pygraphqt'")

        if self.preferred_plot in self._plotting_libraries:
            self._plotting_libraries.remove(self.preferred_plot)
            self._plotting_libraries.insert(0, self.preferred_plot)

        return self._plotting_libraries

    def _find_available_plotting_libraries(self):
        try:
            import plotly
            self._plotting_libraries.append(PlottingLibraries.PLOTLY)
        except ImportError:
            pass
        try:
            import matplotlib
            self._plotting_libraries.append(PlottingLibraries.MATPLOTLIB)
        except ImportError:
            pass
        try:
            import pyqtgraph
            self._plotting_libraries.append(PlottingLibraries.PYGRAPHQT)
        except ImportError:
            pass


global_config = Configuration()
