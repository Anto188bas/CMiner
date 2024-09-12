# CMiner

CMiner is an algorithm for mining patterns from graphs using a user-defined support technique. This implementation provides a command-line interface for running the algorithm, with configurable options like minimum and maximum nodes, support, and search approach.

## Installation

### Prerequisites

- Python 3.x
- `pip` (Python package manager)

### Installation steps

1. Clone the repository:
    ```bash
    git clone https://github.com/Anto188bas/CMiner
    cd CMiner
    ```

2. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Install the library in `editable` mode:
    ```bash
    pip install -e .
    ```

### `requirements.txt`
Make sure the `requirements.txt` file lists all necessary dependencies for running CMiner.

## Usage

Once installed, you can run CMiner directly from the command line using:

```bash
CMiner <db_file> <support> [options]
 ```

### Required arguments:
- `db_file`: Absolute path to the graph database file.
- `support`: Minimum support for pattern extraction: values between 0 and 1 for percentage, or specify an absolute value.

### Additional options:
- `-l`, `--min_nodes`: Minimum number of nodes in the pattern (default: 1).
- `-u`, `--max_nodes`: Maximum number of nodes in the pattern (default: infinite).
- `-m`, `--show_mappings`: Display mappings of found patterns (default: 0).
- `-o`, `--output_path`: File path to save results, if not set the results are shown in the console.

### Usage example

Mining patterns from 4 up to 8 nodes, present in at least 50% of graphs in the database.

```bash
CMiner /path/to/the/graph.data 0.5 -l 4 -u 8
 ```

Mining all patterns present in at least 50 graphs in the database.

```bash
CMiner /path/to/the/graph.data 50
 ```

