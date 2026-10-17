FROM pgvector/pgvector:pg16

# Copy the initialization script
COPY schema.sql /docker-entrypoint-initdb.d/

# Expose the default PostgreSQL port
EXPOSE 5432

# Command to run PostgreSQL
CMD ["postgres"]