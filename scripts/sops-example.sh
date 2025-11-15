#!/bin/bash
# SOPS Example: Encrypt/Decrypt sensitive values files

set -e

# Example: Encrypt a values file with AWS KMS
encrypt_values() {
    local values_file=$1
    local kms_key_arn="arn:aws:kms:us-east-1:ACCOUNT_ID:key/KEY_ID"

    sops --encrypt \
        --kms "${kms_key_arn}" \
        --encrypted-regex '^(secrets|postgresql\.auth\.password|jwt\.secretKey)$' \
        "${values_file}" > "${values_file}.enc"

    echo "Encrypted ${values_file} -> ${values_file}.enc"
}

# Example: Decrypt values file
decrypt_values() {
    local encrypted_file=$1

    sops --decrypt "${encrypted_file}"
}

# Example: Edit encrypted values (opens in editor)
edit_encrypted_values() {
    local encrypted_file=$1

    sops "${encrypted_file}"
}

# Usage examples:
# encrypt_values charts/tars/values-production.yaml
# decrypt_values charts/tars/values-production.yaml.enc | helm upgrade tars ./charts/tars -f -
# edit_encrypted_values charts/tars/values-production.yaml.enc
