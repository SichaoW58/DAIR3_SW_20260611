FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace/resources/unit_5

COPY resources/unit_5/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY resources/unit_5 ./

# Default: execute the Unit 5 notebook in place.
CMD ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", "EDA.ipynb"]
