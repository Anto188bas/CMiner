import sys
from   NetworkX.NetworkConfigurator import NetworkConfigurator
from   NetworkX.NetworksLoading     import NetworksLoading


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python CMinerAccessPoint.py <type_file>")
        sys.exit(1)
    type_file    = sys.argv[1]
    configurator = NetworkConfigurator(type_file)
    networks     = NetworksLoading(type_file, configurator.config)

    for network_name, network_obj in networks.Networks.items():
        print(network_name)
        print(len(network_obj.nodes()))
        print(len(network_obj.edges()))
        print("\n\n")