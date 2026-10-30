# etrade (Python reference implementation)

Execution and market-making research library: Almgren-Chriss optimal
execution, Avellaneda-Stoikov market making, execution analytics,
power-law (square-root) impact with a pre-trade report, and conversion from
market units. Modules: `almgren_chriss`, `avellaneda_stoikov`, `analytics`,
`impact`, `market`. This is the reference implementation; the C++, Rust and
Java ports are validated against it through shared golden values.

```bash
pip install -e "python[dev]"     # from the repository root
pytest python                    # run the tests (174)
python python/demo.py            # run the demo
```

Documentation: https://ashjha0.github.io/electronic-trading/
(source: https://github.com/AshJha0/electronic-trading)
