"""Provider factory — returns the configured DataSource implementation."""

from .base import DataSource
from .yfinance_provider import YFinanceProvider

_PROVIDERS: dict[str, type[DataSource]] = {
    "yfinance": YFinanceProvider,
}


def get_data_source(provider: str = "yfinance") -> DataSource:
    """Return a DataSource instance for the given provider name.

    Parameters
    ----------
    provider : str
        Provider key. Currently only "yfinance" is supported.

    Returns
    -------
    DataSource
        Instantiated provider.
    """
    if provider not in _PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'. Available: {list(_PROVIDERS)}")
    return _PROVIDERS[provider]()
