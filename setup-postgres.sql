-- ==========================================================================
-- ECG Anomaly Detection — PostgreSQL Setup Script
-- ==========================================================================
-- Run this once to create the database + user.
--
--   psql -U postgres -f setup-postgres.sql
--
-- (You'll be prompted for the postgres user's password.)
-- ==========================================================================

-- Create the application user (change password if needed)
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'ecg_user') THEN
        CREATE ROLE ecg_user WITH LOGIN PASSWORD 'ecg_password';
    END IF;
END
$$;

-- Create the application database (owned by ecg_user)
SELECT 'CREATE DATABASE ecg_anomaly OWNER ecg_user'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'ecg_anomaly')\gexec

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE ecg_anomaly TO ecg_user;

-- Connect to the new database and grant schema permissions
\c ecg_anomaly
GRANT ALL ON SCHEMA public TO ecg_user;

\echo '=================================================='
\echo ' PostgreSQL setup complete!'
\echo '=================================================='
\echo ' Database: ecg_anomaly'
\echo ' User:     ecg_user'
\echo ' Password: ecg_password'
\echo ''
\echo ' Now update your .env file with:'
\echo '   DATABASE_URL="postgresql://ecg_user:ecg_password@localhost:5432/ecg_anomaly"'
\echo '=================================================='
