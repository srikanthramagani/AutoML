# Database Architecture

PostgreSQL is the intended metadata store. Phase 1 stores dataset, training-job, and model metadata only; CSV and `.pkl` binaries remain in filesystem or object storage. SQLite is the default development fallback through `DATABASE_URL`.
