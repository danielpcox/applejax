# Copilot Code Review Instructions

JAX PJRT plugin for Apple Metal Performance Shaders. Review PRs against these requirements.

## Review Philosophy

**Assume CI passes.** PRs are reviewed after tests pass. Do not comment that code "will fail" or "will raise"—if tests pass, it works. Do not suggest fixes for problems that don't exist.

Prioritize high-impact feedback. Focus review effort on:

- Bugs, correctness issues, and security vulnerabilities
- Architectural violations (see guidelines below)
- Missing tests for new functionality
- Performance problems with measurable impact

Avoid commenting on:

- Speculative future scenarios that aren't part of the current change
- Availability of standard Unix tools (bc, awk, grep) in developer scripts
- Constants whose meaning is clear from context
- Alternative implementations when the current code is correct and readable
- Syntax or type errors and code formatting or style (will be caught by the compiler and linters)
- API usage that you believe might fail at runtime—the test suite catches these issues
- Type annotations that you think might be incorrect—pyright catches these in CI
- Code that you speculate "will raise" or "won't work"—if CI passes, it works
- Third-party API usage (JAX, NumPy, pytest)—do not claim an API call is wrong unless you are certain

## Architecture

- Reject ops that modify core files. New ops must be self-contained in `src/pjrt_plugin/ops/`.
- Core files (flag if touched by op PRs):
  - `mps_executable.{h,mm}`
  - `stablehlo_parser.{h,mm}`
  - `pjrt_*.{h,cc}`
  - `mps_client.{h,mm}`, `mps_device.{h,mm}`, `mps_buffer.{h,mm}`

## Naming Conventions

- Use PascalCase for handler functions (Objective-C convention).

```cpp
// Correct
ProcessResult HandleDotGeneral(...);
ProcessResult HandleTopK(...);

// Incorrect - flag these
ProcessResult handleDotGeneral(...);  // camelCase
ProcessResult handle_dot_general(...);  // snake_case
```

## Code Style

- Use existing registration macros. Do not create new registries.

```objc
// Correct - simple unary op
REGISTER_MLIR_UNARY_OP("stablehlo.tanh", tanh, Tanh);

// Correct - complex handler
REGISTER_MPS_OP("stablehlo.dot_general", HandleDotGeneral);

// Incorrect - new registry
static std::map<std::string, Handler> myCustomRegistry;  // Flag this
```

- Use `type_utils.{h,mm}` for type conversions. Do not hand-roll conversions.

```cpp
// Correct
MPSDataType mpsType = MlirTypeToMps(mlirType);

// Incorrect - flag manual conversion
MPSDataType mpsType;
if (mlirType.isF32()) mpsType = MPSDataTypeFloat32;  // Use type_utils instead
```

- Use `GetInputTensor(ctx, index)` and `Result(ctx, tensor, "name")` helpers.

```cpp
// Correct
MPSGraphTensor* input = GetInputTensor(ctx, 0);
return Result(ctx, output, "my_op");

// Incorrect - manual context access
auto* input = ctx.values[op->getOperand(0).getAsOpaquePointer()];  // Use GetInputTensor
```

## Testing Guidelines

- Every op requires an `OperationTestConfig` in `tests/configs/`.
- Flag PRs adding ops without corresponding test configs.
- Flag any `@pytest.mark.skip` or `@pytest.mark.xfail` without maintainer approval.

```python
# Required pattern for each op — positional args, factory functions
OperationTestConfig(
    jnp.tanh,
    lambda key: random.normal(key, (3, 4)),
)
```

- `differentiable_argnums=()` is only valid when no arguments are truly differentiable (e.g., integer/boolean ops, or patterns where the gradient generates unsupported scatter/control flow). Flag uses that skip gradient tests without justification.

```python
# Correct - bitwise ops on integers have no gradients
OperationTestConfig(jnp.bitwise_and, ..., differentiable_argnums=())

# Correct - gradient generates unsupported scatter pattern on MPS
OperationTestConfig(lambda x, v: x.at[1:-1].set(v), ..., differentiable_argnums=())

# Incorrect - tanh has differentiable float input, must test gradients
OperationTestConfig(jnp.tanh, ..., differentiable_argnums=())  # Flag this
```

- Test configs should exercise the op's behavior, not just prove it runs. Flag configs that only test a single shape or ignore the op's parameters. If an op takes an `axis` argument, test different axes. If it supports batching, test batched inputs. If it has configuration kwargs, vary them.

## Review Checklist

- Changes confined to `src/pjrt_plugin/ops/` and `tests/configs/`
- Handler uses PascalCase
- Uses `REGISTER_*` macros, no new registries
- Uses `GetInputTensor(ctx, i)` and `Result(ctx, tensor, "name")` helpers
- Test config present in `tests/configs/`
- No modifications to core infrastructure files
- No unexplained skip/xfail markers
- `differentiable_argnums=()` only for truly non-differentiable ops
- Test configs exercise the op's parameters and edge cases, not just a single happy path
