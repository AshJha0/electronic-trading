# Publishing the documentation site (GitHub Pages)

The site at https://ashjha0.github.io/electronic-trading/ is built with
[MkDocs Material](https://squidfunk.github.io/mkdocs-material/) from the
Markdown files in this repository. Each document has one source:

| page | source |
|---|---|
| Home | `docs/site/pages/index.md` |
| Learn | `LEARN.md` |
| Cookbook | `COOKBOOK.md` |
| API contract | `API_SPEC.md` |
| Architecture | `docs/ARCHITECTURE.md` |
| Diagrams | `docs/site/pages/diagrams.md`, which includes `docs/diagrams/*.mmd` |
| Changelog | `CHANGELOG.md` |

`mkdocs.yml` holds the configuration; `docs/site/hooks.py` adds the root
Markdown files to the build so they are not copied. LaTeX in LEARN is
rendered by MathJax and the Mermaid diagrams by Material's built-in Mermaid
support.

## Build locally

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r docs/site/requirements.txt
mkdocs serve                                      # http://127.0.0.1:8000/electronic-trading/
mkdocs build --strict                             # what CI runs
```

## How it is published

`.github/workflows/docs.yml` runs `mkdocs build --strict` on every pull
request and on every push to `main`. On `main` it also deploys the built
site with `actions/upload-pages-artifact` and `actions/deploy-pages`, but
only when the repository's Pages source is **GitHub Actions**; the workflow
reads the source from the Pages API and skips the deploy otherwise.

### Switching from the old setup (one-time)

Until now Pages served `docs/index.html` from the `main` branch `/docs`
folder. That keeps working, unchanged, until the source is switched:

1. Repository **Settings → Pages → Build and deployment → Source:
   GitHub Actions**.
2. Re-run the latest **Docs** workflow on `main` (Actions tab → Docs → Run
   workflow), or push any commit to `main`.

After the switch `docs/index.html` is no longer served and can be deleted;
its content is the Home page above.

## Keeping it accurate

The Home page quotes test counts, golden-case counts and demo output that
also appear in `README.md` (and, until it is retired, in `docs/index.html`).
When any of them changes, update all places in the same commit and check
the numbers against a fresh run of the four demos and test suites.
