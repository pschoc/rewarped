# Packaging

1. Update `VERSION` and `CHANGELOG.md`.

2. Check that package can be built and installed.
```bash
python -m pip install build
python -m build
python -m pip install --upgrade dist/rewarped-<version>-<platform-tag>.whl
python -m rewarped.tests
```

3. Merge into `main`.

4. Tag release with `vX.Y.Z`. 

5. Push release commit and tag to GitHub.

6. Check action builds and uploads the package to PyPI.

7. Create a new release on GitHub with a tag and title of `vX.Y.Z`. Use the changelog updates as the description.
