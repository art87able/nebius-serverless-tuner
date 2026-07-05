#!/bin/sh
# Entrypoint for running the tuner inside a Nebius Serverless Job.
# The Job has no browser, so the nebius CLI authenticates from an IAM token
# passed in via the environment (mint one locally with `nebius iam get-access-token`).
set -eu

if [ -n "${NEBIUS_IAM_TOKEN:-}" ]; then
    umask 077
    printf '%s' "$NEBIUS_IAM_TOKEN" > /tmp/nebius-iam-token
    nebius profile create job \
        --endpoint "${NEBIUS_ENDPOINT:-api.nebius.cloud}" \
        --token-file /tmp/nebius-iam-token \
        --parent-id "$NEBIUS_PARENT_ID" >/dev/null
fi

exec tuner "$@"
