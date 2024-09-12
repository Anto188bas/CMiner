# CMiner

CMiner is an algorithm for mining patterns from graphs using a user-defined support technique. This implementation provides a command-line interface for running the algorithm, with configurable options like minimum and maximum nodes, support, and search approach.

## Installation

### Prerequisites

- Python 3.x
- `pip` (Python package manager)

### Installation steps

1. Clone the repository:
    ```bash
    git clone https://github.com/your/repository.git
    cd repository
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
CMiner <db_file> -s <support> [options]
