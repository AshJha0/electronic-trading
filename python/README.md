# etrade (Python reference implementation)

Execution and market-making research library: Almgren-Chriss optimal
execution, Avellaneda-Stoikov market making and execution analytics. This is
the reference implementation; the C++, Rust and Java ports are validated
against it through shared golden values.

```bash
pip install -e "python[dev]"     # from the repository root
pytest python                    # run the tests
python python/demo.py            # run the demo
```

Full documentation: https://github.com/AshJha0/electronic-trading
