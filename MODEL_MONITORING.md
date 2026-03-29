# Model Monitoring (Accuracy & Relevance)

- **Evaluation set**: Collect real match Q&A (scorecards, commentary, summaries) + edge cases; add high-quality Llama-3.3 synthetic Q/A then human-verify.

- **Metrics**: Track Exact Match for facts, ROUGE-L for summaries, numeric accuracy for scores/wickets, retrieval, latency and hallucination rate.

- **User feedback**: Capture thumbs up/down + optional short text, save (query, context, model-version, timestamp) to a small DB, and surface negatives to a review queue.

- **Drift detection**: Monitor unseen entity frequency (new player/team names), embedding-distribution shifts (cosine distance on recent queries), and sudden drops in accuracy/feedback rates; sample recent failures for labeling.

- **Tooling** TensorRT for models, Redis for feedback storage, Prometheus for monitoring, Milvus for retrieval.