"""v1.1 baseline schema

Creates every table that existed in version 1.1. Tables that are already present
(databases created by v1.0/v1.1 with ``create_all``) are skipped, so the same
revision brings fresh, v1.0 and v1.1 databases to the same baseline.

Revision ID: 0001
Revises: (none)
Create Date: 2026-10-13
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = '0001'
down_revision = None
branch_labels = None
depends_on = None


def _missing(table: str) -> bool:
    return not sa.inspect(op.get_bind()).has_table(table)


def upgrade() -> None:
    if _missing('app_settings'):
        op.create_table('app_settings',
        sa.Column('key', sa.String(length=64), nullable=False),
        sa.Column('value', sa.JSON(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('key')
        )
    if _missing('meta'):
        op.create_table('meta',
        sa.Column('key', sa.String(length=64), nullable=False),
        sa.Column('value', sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint('key')
        )
    if _missing('ml_training_runs'):
        op.create_table('ml_training_runs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('dataset_id', sa.String(length=100), nullable=False),
        sa.Column('dataset_version', sa.String(length=32), nullable=False),
        sa.Column('dataset_hash', sa.String(length=64), nullable=False),
        sa.Column('selected_model', sa.String(length=100), nullable=False),
        sa.Column('metrics', sa.JSON(), nullable=False),
        sa.Column('artifact_path', sa.String(length=500), nullable=False),
        sa.Column('mlflow_run_id', sa.String(length=64), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
        )
    if _missing('users'):
        op.create_table('users',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('username', sa.String(length=64), nullable=False),
        sa.Column('password_hash', sa.String(length=255), nullable=True),
        sa.Column('is_admin', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
        )
        with op.batch_alter_table('users', schema=None) as batch_op:
            batch_op.create_index(batch_op.f('ix_users_username'), ['username'], unique=True)

    if _missing('agent_runs'):
        op.create_table('agent_runs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('owner_id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=200), nullable=False),
        sa.Column('suite_id', sa.String(length=100), nullable=False),
        sa.Column('suite_version', sa.String(length=32), nullable=False),
        sa.Column('suite_hash', sa.String(length=64), nullable=False),
        sa.Column('suite_snapshot', sa.JSON(), nullable=False),
        sa.Column('models', sa.JSON(), nullable=False),
        sa.Column('config', sa.JSON(), nullable=False),
        sa.Column('status', sa.String(length=16), nullable=False),
        sa.Column('progress_done', sa.Integer(), nullable=False),
        sa.Column('progress_total', sa.Integer(), nullable=False),
        sa.Column('current_item', sa.String(length=300), nullable=True),
        sa.Column('error', sa.Text(), nullable=True),
        sa.Column('environment', sa.JSON(), nullable=True),
        sa.Column('model_info', sa.JSON(), nullable=True),
        sa.Column('mlflow_run_id', sa.String(length=64), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('finished_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['owner_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
        )
        with op.batch_alter_table('agent_runs', schema=None) as batch_op:
            batch_op.create_index(batch_op.f('ix_agent_runs_owner_id'), ['owner_id'], unique=False)

    if _missing('agent_suites'):
        op.create_table('agent_suites',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('owner_id', sa.Integer(), nullable=False),
        sa.Column('suite_id', sa.String(length=100), nullable=False),
        sa.Column('name', sa.String(length=200), nullable=False),
        sa.Column('version', sa.String(length=32), nullable=False),
        sa.Column('sha256', sa.String(length=64), nullable=False),
        sa.Column('data', sa.JSON(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['owner_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('owner_id', 'suite_id', name='uq_agent_suite_owner')
        )
        with op.batch_alter_table('agent_suites', schema=None) as batch_op:
            batch_op.create_index(batch_op.f('ix_agent_suites_owner_id'), ['owner_id'], unique=False)

    if _missing('api_tokens'):
        op.create_table('api_tokens',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('token_hash', sa.String(length=64), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('expires_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
        )
        with op.batch_alter_table('api_tokens', schema=None) as batch_op:
            batch_op.create_index(batch_op.f('ix_api_tokens_token_hash'), ['token_hash'], unique=True)
            batch_op.create_index(batch_op.f('ix_api_tokens_user_id'), ['user_id'], unique=False)

    if _missing('arena_datasets'):
        op.create_table('arena_datasets',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('owner_id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=200), nullable=False),
        sa.Column('kind', sa.String(length=16), nullable=False),
        sa.Column('task', sa.String(length=16), nullable=False),
        sa.Column('target', sa.String(length=200), nullable=True),
        sa.Column('text_column', sa.String(length=200), nullable=True),
        sa.Column('columns', sa.JSON(), nullable=False),
        sa.Column('labels', sa.JSON(), nullable=False),
        sa.Column('n_rows', sa.Integer(), nullable=False),
        sa.Column('n_train', sa.Integer(), nullable=False),
        sa.Column('n_test', sa.Integer(), nullable=False),
        sa.Column('test_size', sa.Double(), nullable=False),
        sa.Column('seed', sa.Integer(), nullable=False),
        sa.Column('split_source', sa.String(length=16), nullable=False),
        sa.Column('filename', sa.String(length=255), nullable=False),
        sa.Column('size_bytes', sa.Integer(), nullable=False),
        sa.Column('sha256', sa.String(length=64), nullable=False),
        sa.Column('profile', sa.JSON(), nullable=False),
        sa.Column('warnings', sa.JSON(), nullable=False),
        sa.Column('path', sa.String(length=500), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['owner_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
        )
        with op.batch_alter_table('arena_datasets', schema=None) as batch_op:
            batch_op.create_index(batch_op.f('ix_arena_datasets_owner_id'), ['owner_id'], unique=False)

    if _missing('arena_models'):
        op.create_table('arena_models',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('owner_id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=200), nullable=False),
        sa.Column('filename', sa.String(length=255), nullable=False),
        sa.Column('format', sa.String(length=16), nullable=False),
        sa.Column('kind', sa.String(length=16), nullable=False),
        sa.Column('task', sa.String(length=16), nullable=False),
        sa.Column('role', sa.String(length=24), nullable=False),
        sa.Column('size_bytes', sa.Integer(), nullable=False),
        sa.Column('sha256', sa.String(length=64), nullable=False),
        sa.Column('config', sa.JSON(), nullable=False),
        sa.Column('inspection', sa.JSON(), nullable=False),
        sa.Column('trusted', sa.Boolean(), nullable=False),
        sa.Column('path', sa.String(length=500), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['owner_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
        )
        with op.batch_alter_table('arena_models', schema=None) as batch_op:
            batch_op.create_index(batch_op.f('ix_arena_models_owner_id'), ['owner_id'], unique=False)

    if _missing('benchmark_runs'):
        op.create_table('benchmark_runs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('owner_id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=200), nullable=False),
        sa.Column('suite_id', sa.String(length=100), nullable=False),
        sa.Column('suite_version', sa.String(length=32), nullable=False),
        sa.Column('suite_hash', sa.String(length=64), nullable=False),
        sa.Column('models', sa.JSON(), nullable=False),
        sa.Column('config', sa.JSON(), nullable=False),
        sa.Column('status', sa.String(length=16), nullable=False),
        sa.Column('progress_done', sa.Integer(), nullable=False),
        sa.Column('progress_total', sa.Integer(), nullable=False),
        sa.Column('current_item', sa.String(length=300), nullable=True),
        sa.Column('error', sa.Text(), nullable=True),
        sa.Column('environment', sa.JSON(), nullable=True),
        sa.Column('model_info', sa.JSON(), nullable=True),
        sa.Column('mlflow_run_id', sa.String(length=64), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('finished_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['owner_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
        )
        with op.batch_alter_table('benchmark_runs', schema=None) as batch_op:
            batch_op.create_index(batch_op.f('ix_benchmark_runs_owner_id'), ['owner_id'], unique=False)

    if _missing('conversations'):
        op.create_table('conversations',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('owner_id', sa.Integer(), nullable=False),
        sa.Column('title', sa.String(length=200), nullable=False),
        sa.Column('model', sa.String(length=200), nullable=False),
        sa.Column('system_prompt', sa.Text(), nullable=False),
        sa.Column('options', sa.JSON(), nullable=False),
        sa.Column('use_documents', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['owner_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
        )
        with op.batch_alter_table('conversations', schema=None) as batch_op:
            batch_op.create_index(batch_op.f('ix_conversations_owner_id'), ['owner_id'], unique=False)

    if _missing('documents'):
        op.create_table('documents',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('owner_id', sa.Integer(), nullable=False),
        sa.Column('filename', sa.String(length=200), nullable=False),
        sa.Column('stored_name', sa.String(length=80), nullable=False),
        sa.Column('file_type', sa.String(length=8), nullable=False),
        sa.Column('size_bytes', sa.Integer(), nullable=False),
        sa.Column('sha256', sa.String(length=64), nullable=False),
        sa.Column('status', sa.String(length=16), nullable=False),
        sa.Column('error', sa.Text(), nullable=True),
        sa.Column('page_count', sa.Integer(), nullable=False),
        sa.Column('chunk_count', sa.Integer(), nullable=False),
        sa.Column('char_count', sa.Integer(), nullable=False),
        sa.Column('flagged_chunks', sa.Integer(), nullable=False),
        sa.Column('embedding_backend', sa.String(length=32), nullable=True),
        sa.Column('embedding_model', sa.String(length=200), nullable=True),
        sa.Column('embedding_dim', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('processed_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['owner_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
        )
        with op.batch_alter_table('documents', schema=None) as batch_op:
            batch_op.create_index(batch_op.f('ix_documents_owner_id'), ['owner_id'], unique=False)
            batch_op.create_index(batch_op.f('ix_documents_sha256'), ['sha256'], unique=False)

    if _missing('agent_results'):
        op.create_table('agent_results',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('run_id', sa.Integer(), nullable=False),
        sa.Column('model', sa.String(length=200), nullable=False),
        sa.Column('task_id', sa.String(length=100), nullable=False),
        sa.Column('category', sa.String(length=50), nullable=False),
        sa.Column('repeat_index', sa.Integer(), nullable=False),
        sa.Column('status', sa.String(length=16), nullable=False),
        sa.Column('mode', sa.String(length=16), nullable=False),
        sa.Column('success', sa.Boolean(), nullable=False),
        sa.Column('score', sa.Float(), nullable=False),
        sa.Column('checks', sa.JSON(), nullable=True),
        sa.Column('steps', sa.Integer(), nullable=False),
        sa.Column('tool_calls', sa.Integer(), nullable=False),
        sa.Column('invalid_calls', sa.Integer(), nullable=False),
        sa.Column('format_errors', sa.Integer(), nullable=False),
        sa.Column('stop_reason', sa.String(length=32), nullable=True),
        sa.Column('final_answer', sa.Text(), nullable=False),
        sa.Column('trace', sa.JSON(), nullable=True),
        sa.Column('total_ms', sa.Float(), nullable=True),
        sa.Column('prompt_tokens', sa.Integer(), nullable=True),
        sa.Column('output_tokens', sa.Integer(), nullable=True),
        sa.Column('tokens_per_s', sa.Float(), nullable=True),
        sa.Column('peak_ram_used_mb', sa.Float(), nullable=True),
        sa.Column('error', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['run_id'], ['agent_runs.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
        )
        with op.batch_alter_table('agent_results', schema=None) as batch_op:
            batch_op.create_index(batch_op.f('ix_agent_results_model'), ['model'], unique=False)
            batch_op.create_index(batch_op.f('ix_agent_results_run_id'), ['run_id'], unique=False)

    if _missing('arena_runs'):
        op.create_table('arena_runs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('owner_id', sa.Integer(), nullable=False),
        sa.Column('dataset_id', sa.Integer(), nullable=True),
        sa.Column('name', sa.String(length=200), nullable=False),
        sa.Column('dataset_info', sa.JSON(), nullable=False),
        sa.Column('config', sa.JSON(), nullable=False),
        sa.Column('contenders', sa.JSON(), nullable=False),
        sa.Column('status', sa.String(length=16), nullable=False),
        sa.Column('progress_done', sa.Integer(), nullable=False),
        sa.Column('progress_total', sa.Integer(), nullable=False),
        sa.Column('current_item', sa.String(length=300), nullable=True),
        sa.Column('error', sa.Text(), nullable=True),
        sa.Column('environment', sa.JSON(), nullable=True),
        sa.Column('summary', sa.JSON(), nullable=True),
        sa.Column('best_model_path', sa.String(length=500), nullable=True),
        sa.Column('mlflow_run_id', sa.String(length=64), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('finished_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['dataset_id'], ['arena_datasets.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['owner_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
        )
        with op.batch_alter_table('arena_runs', schema=None) as batch_op:
            batch_op.create_index(batch_op.f('ix_arena_runs_owner_id'), ['owner_id'], unique=False)

    if _missing('benchmark_results'):
        op.create_table('benchmark_results',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('run_id', sa.Integer(), nullable=False),
        sa.Column('model', sa.String(length=200), nullable=False),
        sa.Column('case_id', sa.String(length=100), nullable=False),
        sa.Column('category', sa.String(length=50), nullable=False),
        sa.Column('repeat_index', sa.Integer(), nullable=False),
        sa.Column('status', sa.String(length=16), nullable=False),
        sa.Column('response', sa.Text(), nullable=False),
        sa.Column('thinking_chars', sa.Integer(), nullable=False),
        sa.Column('score', sa.Float(), nullable=False),
        sa.Column('passed', sa.Boolean(), nullable=False),
        sa.Column('eval_details', sa.JSON(), nullable=True),
        sa.Column('ttft_ms', sa.Float(), nullable=True),
        sa.Column('total_ms', sa.Float(), nullable=True),
        sa.Column('load_ms', sa.Float(), nullable=True),
        sa.Column('prompt_tokens', sa.Integer(), nullable=True),
        sa.Column('output_tokens', sa.Integer(), nullable=True),
        sa.Column('tokens_per_s', sa.Float(), nullable=True),
        sa.Column('tokens_source', sa.String(length=16), nullable=False),
        sa.Column('peak_ram_used_mb', sa.Float(), nullable=True),
        sa.Column('avg_cpu_pct', sa.Float(), nullable=True),
        sa.Column('runtime_rss_mb', sa.Float(), nullable=True),
        sa.Column('model_mem_mb', sa.Float(), nullable=True),
        sa.Column('error', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['run_id'], ['benchmark_runs.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
        )
        with op.batch_alter_table('benchmark_results', schema=None) as batch_op:
            batch_op.create_index(batch_op.f('ix_benchmark_results_model'), ['model'], unique=False)
            batch_op.create_index(batch_op.f('ix_benchmark_results_run_id'), ['run_id'], unique=False)

    if _missing('chunks'):
        op.create_table('chunks',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('document_id', sa.Integer(), nullable=False),
        sa.Column('chunk_index', sa.Integer(), nullable=False),
        sa.Column('page', sa.Integer(), nullable=True),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('embedding', sa.LargeBinary(), nullable=False),
        sa.Column('flagged', sa.Boolean(), nullable=False),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
        )
        with op.batch_alter_table('chunks', schema=None) as batch_op:
            batch_op.create_index(batch_op.f('ix_chunks_document_id'), ['document_id'], unique=False)

    if _missing('messages'):
        op.create_table('messages',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('conversation_id', sa.Integer(), nullable=False),
        sa.Column('role', sa.String(length=16), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('thinking', sa.Text(), nullable=True),
        sa.Column('model', sa.String(length=200), nullable=True),
        sa.Column('status', sa.String(length=16), nullable=False),
        sa.Column('metrics', sa.JSON(), nullable=True),
        sa.Column('sources', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['conversation_id'], ['conversations.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
        )
        with op.batch_alter_table('messages', schema=None) as batch_op:
            batch_op.create_index(batch_op.f('ix_messages_conversation_id'), ['conversation_id'], unique=False)

    if _missing('arena_results'):
        op.create_table('arena_results',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('run_id', sa.Integer(), nullable=False),
        sa.Column('key', sa.String(length=200), nullable=False),
        sa.Column('source', sa.String(length=16), nullable=False),
        sa.Column('name', sa.String(length=300), nullable=False),
        sa.Column('status', sa.String(length=16), nullable=False),
        sa.Column('metrics', sa.JSON(), nullable=True),
        sa.Column('details', sa.JSON(), nullable=True),
        sa.Column('predictions', sa.JSON(), nullable=True),
        sa.Column('evaluated_idx', sa.JSON(), nullable=True),
        sa.Column('error', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['run_id'], ['arena_runs.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
        )
        with op.batch_alter_table('arena_results', schema=None) as batch_op:
            batch_op.create_index(batch_op.f('ix_arena_results_run_id'), ['run_id'], unique=False)


def downgrade() -> None:
    with op.batch_alter_table('arena_results', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_arena_results_run_id'))

    op.drop_table('arena_results')
    with op.batch_alter_table('messages', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_messages_conversation_id'))

    op.drop_table('messages')
    with op.batch_alter_table('chunks', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_chunks_document_id'))

    op.drop_table('chunks')
    with op.batch_alter_table('benchmark_results', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_benchmark_results_run_id'))
        batch_op.drop_index(batch_op.f('ix_benchmark_results_model'))

    op.drop_table('benchmark_results')
    with op.batch_alter_table('arena_runs', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_arena_runs_owner_id'))

    op.drop_table('arena_runs')
    with op.batch_alter_table('agent_results', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_agent_results_run_id'))
        batch_op.drop_index(batch_op.f('ix_agent_results_model'))

    op.drop_table('agent_results')
    with op.batch_alter_table('documents', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_documents_sha256'))
        batch_op.drop_index(batch_op.f('ix_documents_owner_id'))

    op.drop_table('documents')
    with op.batch_alter_table('conversations', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_conversations_owner_id'))

    op.drop_table('conversations')
    with op.batch_alter_table('benchmark_runs', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_benchmark_runs_owner_id'))

    op.drop_table('benchmark_runs')
    with op.batch_alter_table('arena_models', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_arena_models_owner_id'))

    op.drop_table('arena_models')
    with op.batch_alter_table('arena_datasets', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_arena_datasets_owner_id'))

    op.drop_table('arena_datasets')
    with op.batch_alter_table('api_tokens', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_api_tokens_user_id'))
        batch_op.drop_index(batch_op.f('ix_api_tokens_token_hash'))

    op.drop_table('api_tokens')
    with op.batch_alter_table('agent_suites', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_agent_suites_owner_id'))

    op.drop_table('agent_suites')
    with op.batch_alter_table('agent_runs', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_agent_runs_owner_id'))

    op.drop_table('agent_runs')
    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_users_username'))

    op.drop_table('users')
    op.drop_table('ml_training_runs')
    op.drop_table('meta')
    op.drop_table('app_settings')
