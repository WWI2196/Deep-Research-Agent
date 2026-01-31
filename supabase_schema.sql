-- Supabase schema for deep research checkpoints and artifacts

create table if not exists public.deep_research_runs (
  run_id text primary key,
  phase text,
  state jsonb,
  updated_at bigint
);

create table if not exists public.deep_research_checkpoints (
  id bigserial primary key,
  run_id text not null,
  phase text not null,
  state jsonb,
  created_at bigint
);

create index if not exists idx_deep_research_checkpoints_run_id
  on public.deep_research_checkpoints(run_id);

create table if not exists public.deep_research_artifacts (
  id bigserial primary key,
  run_id text not null,
  artifact_type text not null,
  content text,
  metadata jsonb,
  created_at bigint
);

create index if not exists idx_deep_research_artifacts_run_id
  on public.deep_research_artifacts(run_id);
