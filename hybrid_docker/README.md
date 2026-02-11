Run in order:

``docker build -t my-model-image .``

``docker network create monitor-net``

``
 docker run -d \
  --name recommender \
  --network monitor-net \
  -p 8080:8080 \
  -v /mnt/c/Users/you/Desktop/data:/app/data \
  my-model-image
``

`` docker run -d \
  --name prometheus \
  --network monitor-net \
  -p 9090:9090 \
  -v /mnt/c/Users/you/Desktop/hybrid_docker/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus ``

Invoke the endpoint

``curl "http://localhost:8080/recommend/0001?top_k=20"``

``curl http://localhost:8080/metrics``
