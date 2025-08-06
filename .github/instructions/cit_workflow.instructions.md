---
applyTo: '**'
---
# GitHub Copilot Instructions for Continuous Integration Testing Workflow

## Core Principles
- **Cell-Centric Development**: All development revolves around Jupyter notebook cells as structural nodes
- **Incremental Complexity**: Build solutions using snowball methodology, starting simple and evolving complexity
- **Validation-First**: Every cell execution must be validated against expected results before proceeding
- **Context-Aware**: Always analyze full notebook context before making changes

## Tool Usage Standards

### Notebook Analysis Tools
- **ALWAYS** use `copilot_getNotebookSummary` before any cell operations to understand current state
- Use `read_notebook_cell_output` to analyze execution results with higher token limits
- Use `semantic_search` to find relevant context across the workspace before implementation

### Cell Management Rules
- **CREATE**: Use `edit_notebook_file` with `editType: "insert"` for new cells
- **EDIT**: Use `edit_notebook_file` with `editType: "edit"` for existing cells with minimum 3-5 lines context
- **DELETE**: Use `edit_notebook_file` with `editType: "delete"` only when explicitly required
- **EXECUTE**: Always use `run_notebook_cell` instead of terminal commands for notebook execution

### Execution Standards
- Execute cells immediately after creation/editing to validate functionality
- Compare actual output with expected results before proceeding to next step
- If execution fails, analyze error and implement fixes within the same cell iteration
- Never proceed to next plan step without successful validation

## Quality Gates
- **Pre-Execution**: Verify cell context and dependencies
- **Post-Execution**: Validate output matches expected results
- **Error Handling**: Implement robust error checking and recovery
- **Documentation**: Include clear markdown cells explaining each development step

## Safety Constraints
- Never delete cells without explicit user request
- Always preserve existing working functionality
- Implement changes incrementally to enable easy rollback
- Maintain clean notebook structure without unnecessary temporary variables

## Code Style Requirements
- Use descriptive variable names that reflect notebook context
- Include type hints where applicable
- Add comments explaining complex logic within cells
- Keep cell content focused on single responsibility
- Use markdown cells for step documentation and expected outcomes

## Environment Considerations
- Target system: Docker image nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
- Available resources: 12 logical cores, NVIDIA GeForce RTX 3070TI
- Optimize code for GPU utilization when applicable
- Consider memory constraints for large data operations
