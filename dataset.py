import kagglehub

# Download latest version
path = kagglehub.dataset_download("andrewkronser/cve-common-vulnerabilities-and-exposures")

print("Path to dataset files:", path)