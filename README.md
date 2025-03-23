# A2 Project

This project provides a Docker-based environment for working with Jupyter notebooks and data analysis.

## Prerequisites

- Docker
- Docker Compose
- Git

## Project Structure

```
.
├── docker/
│   └── Dockerfile
├── data/              # Data directory (mounted from a1-data volume)
├── requirements.txt   # Python dependencies
├── docker-compose.yml # Docker Compose configuration
└── README.md         # This file
```

## Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd a2-project
   ```

2. Build and start the Docker container:
   ```bash
   docker-compose up --build
   ```

3. Access Jupyter Lab:
   - Open your browser and navigate to `http://localhost:8888`
   - Use the token provided in the docker-compose.yml file (default: 'your-secret-token-here')

## Development

- The `a1-data` volume is mounted to `/app/data` in the container
- All Python dependencies are listed in `requirements.txt`

## Git Workflow

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit them:
   ```bash
   git add .
   git commit -m "Description of your changes"
   ```

3. Push your changes:
   ```bash
   git push origin feature/your-feature-name
   ```

## Docker Commands

- Start the container:
  ```bash
  docker-compose up
  ```

- Stop the container:
  ```bash
  docker-compose down
  ```

- Rebuild the container:
  ```bash
  docker-compose up --build
  ```

- View logs:
  ```bash
  docker-compose logs -f
  ```

## Data Management

The project uses a Docker volume named `a1-data` to persist data. This volume is mounted to `/app/data` inside the container.

To manage the volume:
- Create: `docker volume create a1-data`
- Remove: `docker volume rm a1-data`
- List: `docker volume ls` 