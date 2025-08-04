---
applyTo: '**'
---
# Jupyter Notebook Workflow Configuration for GitHub Copilot
This document outlines the workflow for managing Jupyter notebooks using GitHub Copilot, focusing on change management, cell editing, and tool call templates.

## Change Management Protocol
- Sequential Steps for Cell Modifications:
  1. Get notebook summary and verify cell context
  2. Create change plan with clear before/after states
  3. Verify using additional imports with user
  4. Execute changes one cell at a time
  5. Verify changes after each operation by comparing expected content
  6. Log all changes for transparency
  ! Specail Note: Ensure all changes are made sequentially and contextually to avoid conflicts.

## Development Guidelines
- **Cell Context**: Always ensure the cell context is understood before making changes.
- **Validation**: After each change, validate the cell content against expected results to ensure correctness.
- **Keep it tidy**: Dont leave any unnecessary temporary files or variables in memory. Dont spam the notebook with too many cells and comments.
- **Special symbols**: Dont use emojis or special symbols in the code cells. Use them only in markdown cells for documentation purposes.

## System Specifications
- Docker image = nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
- CPU = 12 logical cores
- GPU = NVIDIA GeForce RTX 3070TI

## Set Notebook path based on the current context
```python
notebook_path = "[ABSOLUTE_NOTEBOOK_PATH]"
```
## Tool Call Templates

### 1. Get Notebook Summary
This tool retrieves the current state of the notebook, including cell IDs, content, and metadata. It is essential for understanding the context before making any changes.

```python
<function_calls>
<invoke name="copilot_getNotebookSummary">
<parameter name="filePath">[notebook_path]</parameter>
</invoke>
  * Use copilot_getNotebookSummary to understand current notebook state and cell ordering
  * Store current cell content and metadata for potential rollback
  * Validate cell IDs before operations
```
### 2. Create change plan with clear identification of cells to be modified
This tool helps in planning the changes by identifying the cells that need to be modified and outlining the expected changes.

```python
# Example change plan
change_plan = {
    "cells": [
        {
            "id": "[cell_id]",
            "before": "[original_content_description]",
            "after": "[modified_content_description]"
        },
        {
            "id": "[another_cell_id]",
            "before": "[original_content_description]",
            "after": "[modified_content_description]"
        }
    ]
}
<invoke name="copilot_createChangePlan">
<parameter name="changePlan">change_plan</parameter>
</invoke>
```

### 3. Edit Cell Content
This tool allows for editing the content of a specific cell in the notebook. It requires to verify all additional imports with the user before proceeding.

```python
# 1. Get Notebook Summary
<function_calls>
<invoke name="copilot_getNotebookSummary">
<parameter name="filePath">[notebook_path]</parameter>
</invoke>

# 2. Validate cell IDs and context
<invoke name="copilot_validateCellContext">
<parameter name="cellId">[cell_id]</parameter>
<parameter name="expectedContent">[expected_content]</parameter>
</invoke>

# 3. Edit cell content
<invoke name="copilot_editCell">
<parameter name="cellId">[cell_id]</parameter>
<parameter name="newContent">[new_content]</parameter>
</invoke>

# 4. Verify changes
<invoke name="copilot_verifyCellChange">
<parameter name="cellId">[cell_id]</parameter>
<parameter name="expectedContent">[expected_content]</parameter>
</invoke>

# 5. Log change
<invoke name="copilot_logChange">
<parameter name="cellId">[cell_id]</parameter>
<parameter name="newContent">[new_content]</parameter>
<parameter name="changeDescription">[change_description]</parameter>
</invoke>
```