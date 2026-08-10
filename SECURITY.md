# Security

Please report vulnerabilities privately to the repository maintainers before
opening a public issue. Do not attach checkpoints, SDK archives, board
credentials, proprietary shared objects, mapper dumps or raw production data.

The release builder rejects symlinks, absolute paths, unexpected large files
and known private-artifact suffixes in source archives. Hugging Face tokens are
read only from the environment by the external `hf` process.
