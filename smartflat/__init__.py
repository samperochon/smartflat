# Legacy module alias: pickles created under the thesis-era 'api' package
# reference 'api.*' modules. Redirect them to 'smartflat.*' for deserialization.
import sys as _sys
import importlib as _importlib


class _LegacyApiImporter:
    """Redirect 'api.*' imports to 'smartflat.*' for thesis-era pickle compat."""

    def find_module(self, fullname, path=None):
        if fullname == 'api' or fullname.startswith('api.'):
            return self
        return None

    def load_module(self, fullname):
        if fullname in _sys.modules:
            return _sys.modules[fullname]
        real_name = 'smartflat' + fullname[3:]
        mod = _importlib.import_module(real_name)
        _sys.modules[fullname] = mod
        return mod


_sys.meta_path.insert(0, _LegacyApiImporter())
