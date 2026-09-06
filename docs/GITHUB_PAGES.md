# Publishing the documentation site (GitHub Pages)

This folder contains a self-contained landing page (`index.html`) for
**electronic-trading**, styled identically to the sister projects
([intraday-alpha-platform](https://ashjha0.github.io/intraday-alpha-platform/),
[quant-portfolio](https://ashjha0.github.io/quant-portfolio/)).

## One-time setup

1. Push `main` to GitHub.
2. Repository **Settings → Pages**.
3. Under *Build and deployment*, choose **Source: Deploy from a branch**,
   **Branch: `main`**, **Folder: `/docs`**, then **Save**.
4. After the first build (about a minute) the site is served at
   `https://ashjha0.github.io/electronic-trading/`.

## What is published

- `docs/index.html` — the landing page (links into `LEARN.md`, `COOKBOOK.md`,
  `API_SPEC.md`, `docs/ARCHITECTURE.md` and the diagrams on GitHub).
- Everything else under `docs/` is reachable by its path, e.g.
  `https://ashjha0.github.io/electronic-trading/ARCHITECTURE.md` (served raw).

## Keeping it accurate

The numbers block on the landing page ("Measured, not claimed") quotes the
test counts and golden-case counts stated in `README.md`. When those change,
update both places in the same commit — the page is static HTML and has no
build step.
