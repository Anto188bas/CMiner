import os
import sys
from NetworkX.NetworkConfigurator import NetworkConfigurator
from NetworkX.NetworksLoading import NetworksLoading

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python CMinerAccessPoint.py <data_file>")
        sys.exit(1)

    # Get the data file path from the command line arguments
    data_file = sys.argv[1]

    # Construct the absolute path for the data file
    data_file = os.path.abspath(data_file)

    # Construct the absolute path for the configuration file
    config_file = data_file + '_conf.json'

    configurator = NetworkConfigurator(config_file)
    networks = NetworksLoading(data_file, configurator.config)

    for network_name, network_obj in networks.Networks.items():
        print(network_name)
        print(len(network_obj.nodes()))
        print(len(network_obj.edges()))
        print("\n\n")
