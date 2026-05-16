"""robit.engines — peer-level engines, no plugin namespaces.

Each engine is a self-contained unit implementing PluginAdapter. Discovery
is a glob over `enchanter/engines/*/` looking for `engine.toml` (the manifest)
and an exported `adapter` callable. Phase 1.1 starts with destructive_op_gate.
"""
