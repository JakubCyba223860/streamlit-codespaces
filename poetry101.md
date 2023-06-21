## Working with Poetry Package Manager
Poetry is a popular package manager for Python that simplifies dependency management and project packaging. It provides a streamlined workflow for managing project dependencies, virtual environments, and packaging.

### Installation

To install Poetry, you can use the following command:

```PowerShell
pip install poetry
```
Alternatively, you can visit the Poetry installation page for detailed installation instructions for your specific operating system.

### Managing Dependencies
Poetry uses a `pyproject.toml` file to manage project dependencies.
To add or remove dependencies open a shell ternminal and navigate to the project directory where the `.toml` file is located, this file is always located in the root directory of the project.

You can add dependencies to your project using the `add` command:

```PowerShell
poetry add package_name
```

You can remove dependicies to your project using the `remove` command:

```PowerShell
poetry remove package_name
```

Poetry will automatically resolve and install the specified package and its dependencies. The `pyproject.toml` file will be updated with the added package.

### Managing Virual Enviroments
Poetry creates a dedicated virtual environment for each project, ensuring that dependencies are isolated. To activate the project's virtual environment, use the following command:

```PowerShell
poetry shell
```

This will activate the virtual environment, and you can verify it by checking the command prompt, which should indicate the active virtual environment.