## Application Build and Run using Docker

### Build the Docker Image

```bash
docker build --platform linux/amd64 -t pdf-processor .
```

### Run the Docker Container

```bash
docker run --rm -v $(pwd)/sample_dataset/pdfs:/app/pdfs:ro -v $(pwd)/sample_dataset/outputs:/app/outputs --network none pdf-processor
```

### file structure
docdot/
    .gitignore
    requirements.txt
    Dockerfile
    heading_detector.py
    extractor.py
    process_pdfs.py