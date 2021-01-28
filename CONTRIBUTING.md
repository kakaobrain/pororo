# Contributing to Pororo

## Style check guide

- `pororo` relies on `black` and `isort` to format its source code consistently. After you make changes, format them with:

```bash
$ make style
```

- `pororo` also relies on `yapf` to maintain neat code structure. To apply `yapf`, follow [installation guide](https://github.com/google/yapf#installation) and utilize it with:

```
PYTHONPATH=DIR python DIR/yapf pororo --style '{based_on_style: google, indent_width: 4}' --recursive -i
```

- `pororo` uses `flake8` to check for coding mistakes. You can run the checks with:

```bash
$ make quality
```
