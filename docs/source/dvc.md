## Data Version Control

This project uses ssh-based [dvc](https://dvc.org/) for data versioning. The repository is located on DTU's databar cluster at `/zhome/2f/e/214039/mlops-project` and can be accessed by any DTU user that has ssh access there.

The `./data/` directory is tracked with DVC and excluded from git to keep the repository lightweight while maintaining full data versioning capabilities.

## Prerequisites

### Access to DTU Databar cluster via ssh

1. Establish VPN connection with DTU for key-based ssh connection ([Cisco VPN - ITSwiki](https://itswiki.compute.dtu.dk/index.php/Cisco_VPN)).
2. Create ssh keys for fast access to transfer node ([SSH - gbar.dtu.dk](https://www.gbar.dtu.dk/index.php/faq/53-ssh)).

    ```bash
    $ ssh-keygen -t rsa
    $ ssh-copy-id -i ~/.ssh/id_rsa.pub transfer.gbar.dtu.dk
    $ ssh-copy-id -i ~/.ssh/id_rsa.pub sXXXXXX@transfer.gbar.dtu.dk
    ```
    Replace `sXXXXXX` with your DTU student ID.

3. Verify connection with `ssh sXXXXXX@transfer.gbar.dtu.dk`

## Initial setup

After cloning the repository, configure your DVC remote user:

```bash
$ dvc remote modify --local storage user sXXXXXX
```

Replace `sXXXXXX` with your DTU student ID. This ensures the proper user is set for the `storage` remote (pointing to the DTU databar cluster) if not specified in your `~/.ssh/config` file.

Then pull the data:

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

### Connection issues

If `dvc pull` or `dvc push` fails with connection errors:

1. Verify your VPN connection to DTU is active
2. Test SSH access: `ssh sXXXXXX@transfer.gbar.dtu.dk`
3. Ensure your user is configured: `dvc remote modify --local storage user sXXXXXX`

### Permission issues

The remote storage at `/zhome/2f/e/214039/mlops-project` has group read/write permissions. If you encounter permission errors, ensure you're a member of `dtu` group.

### Data conflicts

If `dvc pull` reports conflicts, your local changes may conflict with remote data:

```bash
$ dvc status              # Check what's different
$ dvc checkout            # Discard local changes and match DVC-tracked version
$ dvc pull                # Pull remote data
```
