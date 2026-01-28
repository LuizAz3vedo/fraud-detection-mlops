-- Criar banco para MLflow
CREATE DATABASE mlflow;

-- Criar banco para dados de features (opcional)
CREATE DATABASE features;

-- Garantir permissões
GRANT ALL PRIVILEGES ON DATABASE mlflow TO fraud_user;
GRANT ALL PRIVILEGES ON DATABASE features TO fraud_user;
