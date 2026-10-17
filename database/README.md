# Database Setup

This directory contains all database-related files for the fintech project.

## Files

- `schema.sql` - Database schema definition including tables, indexes, and triggers

## Usage

The database is run as a Docker container with no persistence. Data is lost when the container stops.

### Building the Database Image

From inside the `database/` directory, run:

```bash
docker build -t fintech-db .
```

### Running the Database Container

After building the image, you can run the database container from any directory:

```bash
docker run --rm -d -p 5432:5432 \
  -e POSTGRES_DB=fintech \
  -e POSTGRES_USER=myuser \
  -e POSTGRES_PASSWORD=mypassword \
  --name fintech-database fintech-db
```

### Stopping and Cleaning Up

```bash
docker stop fintech-database
```

The `--rm` flag ensures the container is automatically removed when stopped, ensuring no data persists.

Note: The database must be running when executing tests, as they validate the database schema and connectivity.

## Connection Details

The database can be accessed at:
- Host: localhost
- Port: 5432
- Database: fintech
- User: myuser
- Password: mypassword

Connection string: `postgresql://myuser:mypassword@localhost:5432/fintech`