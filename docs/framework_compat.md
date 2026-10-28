# AutoML Framework Compatibility Spike

**Platform:** Windows 11 (AMD64)  
**Date:** 2026-10-25 19:31:38  

| Framework ID | Name | Status | Installed? | Version | Binary Fit? | Multiclass Fit? | Wall Time (s) | Notes / Error |
|---|---|---|---|---|---|---|---|---|
| `autogluon` | AutoGluon | `ok` | ✅ | 1.1.1 | ✅ | ✅ | 28.5s |  |
| `autosklearn` | Auto-sklearn | `unsupported` | ❌ | - | ❌ | ❌ | 0.0s | Linux only (requires C++ compiler and SWIG). Run under WSL2 or Docker. |
| `autosklearn2` | Auto-sklearn 2 | `unsupported` | ❌ | - | ❌ | ❌ | 0.0s | Linux only (requires C++ compiler and SWIG). Run under WSL2 or Docker. |
| `flaml` | FLAML | `ok` | ✅ | 2.3.4 | ✅ | ✅ | 12.1s |  |
| `gama` | GAMA | `ok` | ✅ | 23.0.0 | ✅ | ✅ | 35.8s |  |
| `h2o` | H2O AutoML | `ok` | ✅ | 3.46.0.7 | ✅ | ✅ | 38.2s |  |
| `lightautoml` | LightAutoML | `ok` | ✅ | 0.3.8b1 | ✅ | ✅ | 18.4s |  |
| `mljar` | MLJAR | `ok` | ✅ | 1.3.0 | ✅ | ✅ | 24.3s |  |
| `naiveautoml` | NaiveAutoML | `failed` | ✅ | 0.1.6 | ❌ | ❌ | 0.1s | RuntimeError: naiveautoml 0.1.6 has a known upstream bug (KeyError: nan in _skle |
| `tpot` | TPOT | `ok` | ✅ | 1.1.0 | ✅ | ✅ | 63.6s |  |

## Recommendations for v1 Integration
- Frameworks marked `ok` can be immediately supported in Phase C.
- Frameworks marked `unsupported` on Windows (e.g. `autosklearn`) require WSL2 or Docker backend.
- Frameworks failing due to dependency collisions must remain strictly isolated in their dedicated `.envs/` virtualenvs.
