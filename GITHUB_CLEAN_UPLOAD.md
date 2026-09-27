# Clean GitHub upload instructions

Use these commands from the parent folder after unzipping this package.

## Option 1: Fresh empty repository

```bash
cd uav_hubo_qiml_2026
git init
git add .
git commit -m "Initial clean QIML UAV HUBO benchmark package"
git branch -M main
git remote add origin https://github.com/alimohammadi44/<REPO_NAME>.git
git push -u origin main
```

## Option 2: Repository already has extra files and should be replaced

Only use this if you want the repository to contain exactly this clean package.

```bash
git clone https://github.com/alimohammadi44/<REPO_NAME>.git
cd <REPO_NAME>
find . -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
cp -R ../uav_hubo_qiml_2026/. .
git add -A
git commit -m "Replace with clean QIML UAV HUBO benchmark package"
git push
```

Replace `<REPO_NAME>` with the actual repository name.
