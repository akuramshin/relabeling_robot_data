# Evaluation Pipeline for Libero Tasks

This project provides an evaluation pipeline for assessing the performance of different checkpoints on specified tasks using the Libero framework. The evaluation is based on predefined test cases that include specific task IDs and instructions.

## Project Structure

- **configs/**: Contains configuration files for the evaluation pipeline.
  - **test_cases.json**: Defines the test cases to be evaluated, including `task_id` and `instruction`.

- **src/**: Contains the source code for the evaluation pipeline.
  - **eval_libero.py**: The main script that executes the evaluation process, loading test cases and comparing checkpoint performances.
  - **eval_utils.py**: Utility functions for loading configurations, running evaluations, and saving results.
  - **__init__.py**: Marks the directory as a Python package.

- **results/**: Stores the evaluation results for each test case.
  - **task_name_1.json**: Results for the first test case.
  - **task_name_2.json**: Results for the second test case.

- **README.md**: Documentation for the project, including setup and usage instructions.

- **requirements.txt**: Lists the required Python dependencies for the project.

## Setup Instructions

1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Install the required dependencies using:
   ```
   pip install -r requirements.txt
   ```
4. Modify the `configs/test_cases.json` file to define your test cases.

## Running the Evaluation

To run the evaluation pipeline, execute the following command:
```
python src/eval_libero.py --finetuned_path <path_to_checkpoints>
```
Replace `<path_to_checkpoints>` with the path to your finetuned model checkpoints.

## Test Cases

The test cases are defined in the `configs/test_cases.json` file. Each test case includes:
- `task_id`: The identifier for the task to be evaluated.
- `instruction`: The specific instruction for the task.

## Results

The evaluation results for each test case will be saved in the `results/` directory as JSON files. Each file will contain metrics such as success rate, execution time, and other relevant performance indicators.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.