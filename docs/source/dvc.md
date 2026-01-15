## Data Version Control

This project uses [DVC](https://dvc.org/) for data versioning with Google Cloud Storage as the default remote. No credentials are required to pull data.

The `./data/` directory is tracked with DVC and excluded from git to keep the repository lightweight while maintaining full data versioning capabilities.

## Initial setup

After cloning the repository, pull the data:

```bash
$ dvc pull
```

This will download the versioned data from the remote storage into your `./data/` directory.

## Versioning and contributing

By convention, data changes are tagged in the git history as follows:

```bash
git tag -a "v1.0" -m "data v1.0"
```

### Making changes to data

If changes are introduced to the data, follow this workflow:

```bash
$ dvc add data/           # Track changes to the data directory
$ git add data.dvc .gitignore    # Stage DVC metadata files
$ git commit -m "<change description>"
$ git tag -a "v1.x" -m "data v1.x"  # Tag with an adjusted data version
$ git push --follow-tags         # Push git commits and tags
$ dvc push                # Push data to remote storage
```

**Note:** `uv run` might be necessary if project's virtual environment is not activated. Major version changes should be well-justified.

### Checking data status

To see if your local data differs from what's tracked:

```bash
$ dvc status
```

### Pulling specific data versions

To retrieve data from a specific version tag:

```bash
$ git checkout v1.0
$ dvc pull
```

## Troubleshooting

### Data conflicts

If `dvc pull` reports conflicts, your local changes may conflict with remote data:

```bash
$ dvc status              # Check what's different
$ dvc checkout            # Discard local changes and match DVC-tracked version
$ dvc pull                # Pull remote data
```

## Deprecated: SSH Remote

The legacy SSH-based remote (`storage`) pointing to DTU's databar cluster is still available but deprecated.

### Prerequisites

1. Establish VPN connection with DTU ([Cisco VPN - ITSwiki](https://itswiki.compute.dtu.dk/index.php/Cisco_VPN)).
2. Create SSH keys for the transfer node ([SSH - gbar.dtu.dk](https://www.gbar.dtu.dk/index.php/faq/53-ssh)):

    ```bash
    $ ssh-keygen -t rsa
    $ ssh-copy-id -i ~/.ssh/id_rsa.pub transfer.gbar.dtu.dk
    $ ssh-copy-id -i ~/.ssh/id_rsa.pub sXXXXXX@transfer.gbar.dtu.dk
    ```
    Replace `sXXXXXX` with your DTU student ID.

3. Configure DVC to use the SSH remote:

    ```bash
    $ dvc remote default storage
    $ dvc remote modify --local storage user sXXXXXX
    ```
