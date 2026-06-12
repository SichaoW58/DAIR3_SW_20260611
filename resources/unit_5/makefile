# Variables
NOTEBOOK = EDA.ipynb
OUTPUT_DIR = reports
IMAGE_NAME = dair3-unit5:latest



# Default target
all: run

# Run the notebook and update it in-place
run:
	jupyter nbconvert --to notebook --execute --inplace $(NOTEBOOK)

# Run the notebook and export the results as an HTML report
html:
	mkdir -p $(OUTPUT_DIR)
	jupyter nbconvert --to html --execute $(NOTEBOOK) --output-dir=$(OUTPUT_DIR)

# Build Unit 5 image from repository root context
docker-build:
	cd ../.. && docker build -t $(IMAGE_NAME) -f resources/unit_5/Dockerfile .

# Run notebook execution inside the Unit 5 image
docker-run:
	docker run --rm $(IMAGE_NAME)
