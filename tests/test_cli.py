from click.testing import CliRunner
import pytest

from mafat_radar_challenge.cli import cli


@pytest.fixture()
def runner():
    return CliRunner()


def test_cli_template(runner):
    result = runner.invoke(cli)
    assert result.exit_code == 0
