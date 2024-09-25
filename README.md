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

Once installed, CMiner can be used in three different ways:

1. **Command Line Interface (CLI)**:
    Run directly from the command line with the following syntax:

```bash
CMiner <db_file> <support> [options]
 ```

2. **Using Python's `-m` flag**:
   Alternatively, you can execute CMiner as a Python module:
```bash
python -m CMiner <db_file> <support> [options]
 ```

2. **As a Python module**:
   You can also import CMiner into your Python code and use it programmatically:
   
```python
from CMiner import CMiner

miner = CMiner(
    db_file='/path/to/your/db/graph', # required
    support, # required
    min_nodes=1,
    max_nodes=float('inf'),
    show_mappings=False,
    output_path=None
)
 ```




### Required arguments:
- `db_file`: Absolute path to the graph database file.
- `support`: **Minimum support for pattern extraction**: Specify a value between `0` and `1` for percentage (e.g., `0.2` for 20%) or an absolute number (e.g., `20` for at least 20 graphs).


### Additional options:
- `-l`, `--min_nodes`: Minimum number of nodes in the pattern (default: 1).
- `-u`, `--max_nodes`: Maximum number of nodes in the pattern (default: infinite).
- `-m`, `--show_mappings`: Display mappings of found patterns (default: 0).
- `-o`, `--output_path`: File path to save results, if not set the results are shown in the console.
- `-p`, `--patterns_path`: File paths to start the search.

### Usage example

- Mining patterns from 4 up to 8 nodes, present in at least 50% of graphs in the database.

```bash
CMiner /path/to/the/graph.data 0.5 -l 4 -u 8
 ```

- Mining all patterns present in at least 50 graphs in the database.

```bash
CMiner /path/to/the/graph.data 50
 ```

- Mining all patterns present in at least 50 graphs in the database that have the pattern inside the file `patterns.txt`

```bash
CMiner /path/to/the/graph.data 50 -p patterns.txt
 ```
Content of `patterns.txt`
```bash
v 1 purple
v 2 yellow
e 2 1 white
-
v 1 blue
v 2 yellow
v 3 red
e 1 2 white
e 1 3 white
 ```



