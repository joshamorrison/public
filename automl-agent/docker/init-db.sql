-- Initialize AutoML Agent Platform Database

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    permissions TEXT[] DEFAULT ARRAY['read'],
    api_quota INTEGER DEFAULT 100,
    api_usage INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE
);

-- Create API keys table
CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    key_id VARCHAR(255) UNIQUE NOT NULL,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    permissions TEXT[] DEFAULT ARRAY['read'],
    quota INTEGER DEFAULT 500,
    usage INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_used TIMESTAMP WITH TIME ZONE,
    expires_at TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT TRUE
);

-- Create jobs table
CREATE TABLE IF NOT EXISTS jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id VARCHAR(255) UNIQUE NOT NULL,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    status VARCHAR(50) NOT NULL DEFAULT 'queued',
    progress FLOAT DEFAULT 0.0,
    agent_type VARCHAR(100),
    task_description TEXT,
    parameters JSONB,
    result JSONB,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    estimated_completion TIMESTAMP WITH TIME ZONE
);

-- Create agent communications table
CREATE TABLE IF NOT EXISTS agent_communications (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID REFERENCES jobs(id) ON DELETE CASCADE,
    sender_agent VARCHAR(100) NOT NULL,
    receiver_agent VARCHAR(100),
    message_type VARCHAR(100) NOT NULL,
    content JSONB NOT NULL,
    priority INTEGER DEFAULT 1,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create model registry table
CREATE TABLE IF NOT EXISTS model_registry (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id VARCHAR(255) UNIQUE NOT NULL,
    job_id UUID REFERENCES jobs(id) ON DELETE CASCADE,
    agent_type VARCHAR(100) NOT NULL,
    model_name VARCHAR(255) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    model_type VARCHAR(100),
    performance_metrics JSONB,
    model_artifacts JSONB,
    storage_path TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE
);

-- Create performance metrics table
CREATE TABLE IF NOT EXISTS performance_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID REFERENCES jobs(id) ON DELETE CASCADE,
    agent_type VARCHAR(100) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    metric_metadata JSONB,
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_user_id ON jobs(user_id);
CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at);
CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_active ON api_keys(is_active);
CREATE INDEX IF NOT EXISTS idx_agent_communications_job_id ON agent_communications(job_id);
CREATE INDEX IF NOT EXISTS idx_model_registry_job_id ON model_registry(job_id);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_job_id ON performance_metrics(job_id);

-- Insert default admin user
INSERT INTO users (username, email, hashed_password, permissions, api_quota)
VALUES (
    'admin',
    'admin@automl.com',
    crypt('admin123', gen_salt('bf')),
    ARRAY['read', 'write', 'admin'],
    1000
) ON CONFLICT (username) DO NOTHING;

-- Insert default demo user
INSERT INTO users (username, email, hashed_password, permissions, api_quota)
VALUES (
    'demo',
    'demo@automl.com',
    crypt('demo123', gen_salt('bf')),
    ARRAY['read', 'write'],
    100
) ON CONFLICT (username) DO NOTHING;

-- Create a demo API key
INSERT INTO api_keys (key_id, user_id, permissions, quota)
SELECT 
    'ak_demo_12345',
    u.id,
    ARRAY['read', 'write'],
    500
FROM users u 
WHERE u.username = 'demo'
ON CONFLICT (key_id) DO NOTHING;